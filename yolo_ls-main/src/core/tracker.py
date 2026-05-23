"""ByteTrack 多目标跟踪模块"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


@dataclass
class Track:
    """跟踪目标"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    def update_history(self):
        """更新历史轨迹"""
        self.history.append(self.center)


class KalmanBoxTracker:
    """使用卡尔曼滤波器的单目标跟踪器"""
    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int]):
        # 状态向量: [x, y, s, r, vx, vy, vs]
        # x, y: 中心坐标, s: 面积, r: 宽高比, vx, vy, vs: 速度
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.age = 0

    def _bbox_to_z(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """将 bbox 转换为观测向量 [x, y, s, r]"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2.
        y = y1 + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 1
        return np.array([x, y, s, r]).reshape((4, 1))

    def _z_to_bbox(self, z: np.ndarray) -> Tuple[int, int, int, int]:
        """将观测向量转换为 bbox"""
        x, y, s, r = z.flatten()[:4]
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        return (
            int(x - w / 2),
            int(y - h / 2),
            int(x + w / 2),
            int(y + h / 2)
        )

    def update(self, bbox: Tuple[int, int, int, int]):
        """更新状态"""
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self._bbox_to_z(bbox))

    def predict(self) -> Tuple[int, int, int, int]:
        """预测下一状态"""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._z_to_bbox(self.kf.x)

    def get_state(self) -> Tuple[int, int, int, int]:
        """获取当前状态"""
        return self._z_to_bbox(self.kf.x)


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """计算两组边界框的 IOU 矩阵"""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    iou = wh / (area_test + area_gt - wh + 1e-6)
    return iou


class ByteTracker:
    """ByteTrack 多目标跟踪器"""

    def __init__(self, track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 min_box_area: int = 10):
        """
        初始化跟踪器

        Args:
            track_thresh: 高置信度阈值
            track_buffer: 跟踪缓冲帧数
            match_thresh: 匹配阈值
            min_box_area: 最小边界框面积
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.trackers: List[KalmanBoxTracker] = []
        self.track_info: Dict[int, dict] = {}  # 存储额外信息
        self.frame_count = 0

    def update(self, detections: List) -> List[Track]:
        """
        更新跟踪器

        Args:
            detections: Detection 对象列表

        Returns:
            Track 对象列表
        """
        self.frame_count += 1

        # 提取检测信息
        if len(detections) == 0:
            dets = np.empty((0, 5))
            det_info = []
        else:
            dets = np.array([
                [*d.bbox, d.confidence] for d in detections
            ])
            det_info = [(d.class_id, d.class_name) for d in detections]

        # 分离高低置信度检测
        high_mask = dets[:, 4] >= self.track_thresh if len(dets) > 0 else np.array([])
        high_dets = dets[high_mask] if len(dets) > 0 else np.empty((0, 5))
        low_dets = dets[~high_mask] if len(dets) > 0 else np.empty((0, 5))
        high_info = [det_info[i] for i, m in enumerate(high_mask) if m] if len(dets) > 0 else []
        low_info = [det_info[i] for i, m in enumerate(high_mask) if not m] if len(dets) > 0 else []

        # 预测现有跟踪器
        for trk in self.trackers:
            trk.predict()

        # 第一次关联：高置信度检测与跟踪器
        matched, unmatched_dets, unmatched_trks = self._associate(
            high_dets, self.trackers, self.match_thresh
        )

        # 更新匹配的跟踪器
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(tuple(high_dets[det_idx, :4].astype(int)))
            self.track_info[self.trackers[trk_idx].id] = {
                'class_id': high_info[det_idx][0],
                'class_name': high_info[det_idx][1],
                'confidence': high_dets[det_idx, 4]
            }

        # 第二次关联：低置信度检测与未匹配跟踪器
        if len(low_dets) > 0 and len(unmatched_trks) > 0:
            unmatched_trackers = [self.trackers[i] for i in unmatched_trks]
            matched2, _, remaining_trks = self._associate(
                low_dets, unmatched_trackers, 0.5
            )
            for det_idx, trk_idx in matched2:
                real_trk_idx = unmatched_trks[trk_idx]
                self.trackers[real_trk_idx].update(tuple(low_dets[det_idx, :4].astype(int)))
                self.track_info[self.trackers[real_trk_idx].id] = {
                    'class_id': low_info[det_idx][0],
                    'class_name': low_info[det_idx][1],
                    'confidence': low_dets[det_idx, 4]
                }

        # 为未匹配的高置信度检测创建新跟踪器
        for det_idx in unmatched_dets:
            bbox = tuple(high_dets[det_idx, :4].astype(int))
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > self.min_box_area:
                trk = KalmanBoxTracker(bbox)
                self.trackers.append(trk)
                self.track_info[trk.id] = {
                    'class_id': high_info[det_idx][0],
                    'class_name': high_info[det_idx][1],
                    'confidence': high_dets[det_idx, 4]
                }

        # 移除过期跟踪器
        self.trackers = [
            trk for trk in self.trackers
            if trk.time_since_update <= self.track_buffer
        ]

        # 生成输出
        tracks = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and trk.hits >= 3:
                info = self.track_info.get(trk.id, {})
                track = Track(
                    track_id=trk.id,
                    bbox=trk.get_state(),
                    confidence=info.get('confidence', 0),
                    class_id=info.get('class_id', -1),
                    class_name=info.get('class_name', 'unknown'),
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=trk.time_since_update
                )
                track.update_history()
                tracks.append(track)

        return tracks

    def _associate(self, dets: np.ndarray, trackers: List[KalmanBoxTracker],
                   thresh: float) -> Tuple[List, List, List]:
        """关联检测与跟踪器"""
        if len(trackers) == 0:
            return [], list(range(len(dets))), []
        if len(dets) == 0:
            return [], [], list(range(len(trackers)))

        trk_bboxes = np.array([trk.get_state() for trk in trackers])
        iou_matrix = iou_batch(dets[:, :4], trk_bboxes)

        # 使用匈牙利算法
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)

        matched = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(trackers)))

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= thresh:
                matched.append((row, col))
                unmatched_dets.remove(row)
                unmatched_trks.remove(col)

        return matched, unmatched_dets, unmatched_trks

    def reset(self):
        """重置跟踪器"""
        self.trackers = []
        self.track_info = {}
        self.frame_count = 0
        KalmanBoxTracker.count = 0
