"""
自适应违规检测模块

创新点：智能识别特殊情况
- 检测特种车辆（救护车、消防车、警车）
- 检测交警指挥
- 如果违规时附近有特种车辆或交警，标记为"异常"待人工复核
- 异常情况单独保存截图，便于后续人工审核
"""
import logging
import time
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

from src.utils.bbox import clamp_bbox
from src.utils.constants import (
    COLOR_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_COLOR_THRESHOLD,
    TRAFFIC_LIGHT_MIN_ASPECT,
    TRAFFIC_LIGHT_MAX_ASPECT,
    DEFAULT_LANE_CHANGE_LATERAL_PX,
    DEFAULT_LANE_CHANGE_MIN_SPEED_KMH,
    DEFAULT_LANE_CHANGE_HISTORY_LEN,
    DEFAULT_EMERGENCY_DISTANCE_PX,
)
from src.utils.hsv import bgr_to_hsv_normalized

from .emergency_vehicle import EmergencyVehicleDetector, EmergencyVehicle, EmergencyVehicleType
from .feature import Direction
from .lane_violation import LaneViolationAnalyzer, build_analyzer_from_violation_config
from .traffic_light_cls import TrafficLightClassifier

logger = logging.getLogger(__name__)

# 同一车辆同一违规类型的最短重复记录间隔（秒）
VIOLATION_COOLDOWN_SEC = 30.0


class ViolationType(Enum):
    """违规类型"""
    RED_LIGHT = "red_light"      # 闯红灯
    SPEEDING = "speeding"        # 超速
    WRONG_WAY = "wrong_way"      # 逆行
    ILLEGAL_LANE = "illegal_lane"  # 违规变道


class AnomalyReason(Enum):
    """异常原因"""
    EMERGENCY_VEHICLE = "emergency_vehicle"  # 附近有特种车辆
    TRAFFIC_POLICE = "traffic_police"        # 附近有交警
    SIGNAL_MALFUNCTION = "signal_malfunction"  # 信号灯故障
    NONE = "none"


# 异常原因的中文描述
ANOMALY_DESCRIPTIONS = {
    AnomalyReason.EMERGENCY_VEHICLE: "附近有特种车辆",
    AnomalyReason.TRAFFIC_POLICE: "附近有交警指挥",
    AnomalyReason.SIGNAL_MALFUNCTION: "信号灯故障",
    AnomalyReason.NONE: "无",
}


@dataclass
class ViolationRecord:
    """违规记录"""
    record_id: str                          # 记录ID（时间戳）
    violation_type: ViolationType           # 违规类型
    track_id: int                           # 车辆跟踪ID
    timestamp: datetime                     # 发生时间
    location: Tuple[int, int]               # 位置
    speed: Optional[float] = None           # 速度
    plate_number: Optional[str] = None      # 车牌号
    snapshot_path: Optional[str] = None     # 截图路径
    is_anomaly: bool = False                # 是否为异常（需人工复核）
    anomaly_reason: AnomalyReason = AnomalyReason.NONE  # 异常原因
    nearby_objects: List[str] = field(default_factory=list)  # 附近特殊对象

    # ---- Backward compatibility for older callers ----
    @property
    def is_exempted(self) -> bool:
        return self.is_anomaly

    @is_exempted.setter
    def is_exempted(self, value: bool):
        self.is_anomaly = bool(value)

    @property
    def exemption_reason(self) -> AnomalyReason:
        return self.anomaly_reason

    @exemption_reason.setter
    def exemption_reason(self, value: AnomalyReason):
        self.anomaly_reason = value

    @property
    def nearby_emergency_vehicles(self) -> List[str]:
        return self.nearby_objects

    @nearby_emergency_vehicles.setter
    def nearby_emergency_vehicles(self, value: Optional[List[str]]):
        self.nearby_objects = list(value or [])

    @property
    def exemption_details(self) -> str:
        return ", ".join(self.nearby_objects)

    @exemption_details.setter
    def exemption_details(self, value: Optional[str]):
        if not value:
            self.nearby_objects = []
        else:
            self.nearby_objects = [item.strip() for item in value.split(",") if item.strip()]


# 兼容旧代码
ExemptionReason = AnomalyReason
EXEMPTION_DESCRIPTIONS = ANOMALY_DESCRIPTIONS


def select_best_light_bbox(
    light_dets: List,
    stop_line_y: Optional[int] = None,
    frame_h: int = 720,
    frame_w: int = 1280,
) -> Optional[Tuple[int, int, int, int]]:
    """
    从多个红绿灯检测结果中选择最优的一个。

    优先策略：
    1. 若已知停止线 Y 坐标，选离停止线最近的灯（同方向最近灯）
    2. 否则选画面下半部分且面积最大的灯（通常为当前车道的灯）
    3. 兜底：选置信度最高的
    """
    if not light_dets:
        return None
    if len(light_dets) == 1:
        return light_dets[0].bbox

    def _area(det) -> int:
        x1, y1, x2, y2 = det.bbox
        return max(0, (x2 - x1) * (y2 - y1))

    def _center_y(det) -> float:
        _, y1, _, y2 = det.bbox
        return (y1 + y2) / 2.0

    if stop_line_y is not None:
        # 选中心 Y 坐标离停止线最近的灯
        return min(light_dets, key=lambda d: abs(_center_y(d) - stop_line_y)).bbox

    # 无停止线：优先画面下部 60% 区域内的灯，综合面积和置信度
    lower_half = [d for d in light_dets if _center_y(d) > frame_h * 0.4]
    candidates = lower_half if lower_half else light_dets

    return max(candidates, key=lambda d: _area(d) * d.confidence).bbox


class TrafficPoliceDetector:
    """交警检测器（基于颜色和姿态特征）"""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # 交警制服颜色特征 (HSV) - 深蓝色/黑色
        self.uniform_lower = np.array([
            cfg.get('uniform_hue_low', 100),
            cfg.get('uniform_sat_low', 50),
            cfg.get('uniform_val_low', 30),
        ])
        self.uniform_upper = np.array([cfg.get('uniform_hue_high', 130), 255, 150])
        # 反光背心颜色 - 荧光黄/绿
        self.vest_lower = np.array([
            cfg.get('vest_hue_low', 25),
            cfg.get('vest_sat_low', 100),
            cfg.get('vest_val_low', 100),
        ])
        self.vest_upper = np.array([cfg.get('vest_hue_high', 45), 255, 255])
        self._color_threshold = cfg.get('color_threshold', COLOR_RATIO_THRESHOLD)
        self._uniform_threshold = cfg.get('uniform_threshold', COLOR_RATIO_THRESHOLD * 2)

    def detect(self, frame: np.ndarray,
               person_bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        检测交警

        Args:
            frame: BGR图像
            person_bboxes: 人员边界框列表

        Returns:
            交警边界框列表
        """
        police_bboxes = []

        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            hsv = bgr_to_hsv_normalized(roi)
            total = roi.shape[0] * roi.shape[1]

            vest_mask = cv2.inRange(hsv, self.vest_lower, self.vest_upper)
            vest_ratio = np.sum(vest_mask > 0) / total

            # 检测制服颜色
            uniform_mask = cv2.inRange(hsv, self.uniform_lower, self.uniform_upper)
            uniform_ratio = np.sum(uniform_mask > 0) / total

            # 如果有反光背心或制服特征，认为是交警
            if vest_ratio > self._color_threshold or uniform_ratio > self._uniform_threshold:
                police_bboxes.append(bbox)

        return police_bboxes


class TrafficLightDetector:
    """交通灯状态检测器（CNN 优先，HSV 回退）"""

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # HSV 范围：可通过 YAML traffic_light 节覆盖
        # 默认值已针对过曝 LED 放宽（更低饱和度/明度阈值，更宽色调范围）
        self.red_lower1 = np.array([
            cfg.get('red_hue_low1', 0),
            cfg.get('red_sat_low', 30),
            cfg.get('red_val_low', 40),
        ])
        self.red_upper1 = np.array([cfg.get('red_hue_high1', 15), 255, 255])
        self.red_lower2 = np.array([
            cfg.get('red_hue_low2', 155),
            cfg.get('red_sat_low', 30),
            cfg.get('red_val_low', 40),
        ])
        self.red_upper2 = np.array([cfg.get('red_hue_high2', 180), 255, 255])
        self.green_lower = np.array([
            cfg.get('green_hue_low', 40),
            cfg.get('green_sat_low', 30),
            cfg.get('green_val_low', 40),
        ])
        self.green_upper = np.array([cfg.get('green_hue_high', 90), 255, 255])
        self.yellow_lower = np.array([
            cfg.get('yellow_hue_low', 10),
            cfg.get('yellow_sat_low', 30),
            cfg.get('yellow_val_low', 40),
        ])
        self.yellow_upper = np.array([cfg.get('yellow_hue_high', 45), 255, 255])
        self.state_history = deque(maxlen=cfg.get('malfunction_window', 10) * 3)
        self._color_threshold = cfg.get('color_threshold', TRAFFIC_LIGHT_COLOR_THRESHOLD)
        self._malfunction_window = cfg.get('malfunction_window', 10)
        self._malfunction_unknown_thresh = int(
            self._malfunction_window * cfg.get('malfunction_unknown_ratio', 0.8)
        )
        self._malfunction_change_count = cfg.get('malfunction_change_count', 6)
        # 故障判定消抖
        self._malfunction_latch = 0         # 连续故障帧计数
        self._malfunction_engaged = False   # 故障锁存
        self._malfunction_clear_counter = 0 # 连续正常帧计数（锁存后）

        # CNN 分类器（优先使用，不可用时回退 HSV）
        self._use_model = bool(cfg.get('use_model', False))
        self._cnn_conf_threshold = float(cfg.get('cnn_conf_threshold', 0.6))
        self._cnn_classifier: Optional[TrafficLightClassifier] = None
        if self._use_model:
            model_path = cfg.get('model_path', 'models/traffic_light_cls.pt')
            try:
                self._cnn_classifier = TrafficLightClassifier(model_path)
                if not self._cnn_classifier.is_available:
                    logger.warning(
                        "TrafficLightDetector: CNN model not available, using HSV fallback"
                    )
                    self._cnn_classifier = None
            except Exception as exc:
                logger.warning(
                    "TrafficLightDetector: failed to load CNN model: %s — using HSV fallback", exc
                )
                self._cnn_classifier = None

        # 时序平滑与红灯持续帧数追踪
        self._smooth_window = max(3, int(cfg.get('smooth_window', 5)))
        self._red_frame_counter = 0          # 红灯已持续帧数
        self._min_red_frames = int(cfg.get('min_red_frames', 3))  # 判定闯红灯所需的最小红灯持续帧数
        # CLAHE 预处理开关
        self._use_clahe = bool(cfg.get('use_clahe', True))

    def detect_state(self, frame: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> str:
        """检测交通灯状态（CNN 优先，HSV 回退）"""
        fh, fw = frame.shape[:2]
        clamped = clamp_bbox(bbox, fw, fh)
        if clamped is None:
            return 'unknown'
        x1, y1, x2, y2 = clamped
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'

        # ---- 优先使用 CNN 分类器 ----
        if self._cnn_classifier is not None:
            cnn_state, cnn_conf = self._cnn_classifier.classify(frame, clamped)
            if cnn_state != 'unknown' and cnn_conf >= self._cnn_conf_threshold:
                self.state_history.append(cnn_state)
                return cnn_state
            # CNN 置信度不足，回退 HSV（不记录 unknown 到 history）

        aspect = (x2 - x1) / max(1, (y2 - y1))
        is_vertical = TRAFFIC_LIGHT_MIN_ASPECT <= aspect <= TRAFFIC_LIGHT_MAX_ASPECT
        is_horizontal = (1.0 / TRAFFIC_LIGHT_MAX_ASPECT) <= aspect <= (1.0 / TRAFFIC_LIGHT_MIN_ASPECT)

        if not (is_vertical or is_horizontal):
            return self._color_vote(roi)

        hsv = bgr_to_hsv_normalized(roi, use_clahe=self._use_clahe)
        if is_vertical:
            # 竖灯：上红 / 中黄 / 下绿，每段用亮度峰值定位灯体核心区域
            h = roi.shape[0]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bands = self._split_bands_with_peak(hsv, gray, axis=0, num_bands=3)
        else:
            # 横灯：左红 / 中黄 / 右绿
            w = roi.shape[1]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bands = self._split_bands_with_peak(hsv, gray, axis=1, num_bands=3)

        band_states = []
        for band in bands:
            if band.size == 0:
                continue
            band_states.append(self._dominant_color_on_band(band))

        if band_states:
            from collections import Counter
            state = Counter(band_states).most_common(1)[0][0]
        else:
            state = self._color_vote(roi)

        self.state_history.append(state)
        self._update_red_counter(state)
        return state

    def get_smoothed_state(self) -> str:
        """返回最近 N 帧的多数投票状态，用于闯红灯判定时的时序平滑。"""
        if len(self.state_history) == 0:
            return 'unknown'
        recent = list(self.state_history)[-self._smooth_window:]
        from collections import Counter
        return Counter(recent).most_common(1)[0][0]

    def red_duration_frames(self) -> int:
        """返回红灯已持续帧数（用于确认红灯在越线前已亮起）。"""
        return self._red_frame_counter

    def _update_red_counter(self, state: str) -> None:
        """更新红灯持续帧数计数器。"""
        if state == 'red':
            self._red_frame_counter += 1
        else:
            self._red_frame_counter = 0

    def _split_bands_with_peak(
        self, hsv: np.ndarray, gray: np.ndarray,
        axis: int = 0, num_bands: int = 3, peak_margin: float = 0.15,
    ) -> list:
        """
        沿指定轴将 ROI 分为 num_bands 段，每段内用亮度峰值定位灯体核心区域，
        避免粗暴三等分将灯杆/外壳误判为灯体。

        Args:
            hsv: HSV 颜色空间 ROI
            gray: 灰度 ROI（用于亮度峰值检测）
            axis: 0=垂直分带（竖灯），1=水平分带（横灯）
            num_bands: 分段数（默认 3：红/黄/绿）
            peak_margin: 峰值周围保留的边距比例

        Returns:
            HSV band 切片列表
        """
        length = hsv.shape[axis]
        band_size = length // num_bands
        if band_size < 2:
            # ROI 太小，回退到简单均分
            bands = []
            for i in range(num_bands):
                start = i * max(1, length // num_bands)
                end = (i + 1) * max(1, length // num_bands) if i < num_bands - 1 else length
                if axis == 0:
                    bands.append(hsv[start:max(start + 1, end), :])
                else:
                    bands.append(hsv[:, start:max(start + 1, end)])
            return bands

        bands = []
        # 沿 axis 计算亮度投影
        proj = gray.mean(axis=1 - axis) if axis == 0 else gray.mean(axis=0)

        for i in range(num_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < num_bands - 1 else length
            segment = proj[start:end]
            if segment.size == 0:
                continue
            peak_offset = int(np.argmax(segment))
            peak_center = start + peak_offset
            half = max(2, int(band_size * peak_margin))
            crop_start = max(start, peak_center - half)
            crop_end = min(end, peak_center + half + 1)
            if crop_end <= crop_start:
                crop_start, crop_end = start, end
            if axis == 0:
                bands.append(hsv[crop_start:crop_end, :])
            else:
                bands.append(hsv[:, crop_start:crop_end])

        return bands

    def _dominant_color_on_band(self, hsv_band: np.ndarray) -> str:
        total = hsv_band.shape[0] * hsv_band.shape[1]
        if total <= 0:
            return 'unknown'
        red_mask = (
            cv2.inRange(hsv_band, self.red_lower1, self.red_upper1)
            | cv2.inRange(hsv_band, self.red_lower2, self.red_upper2)
        )
        green_mask = cv2.inRange(hsv_band, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv_band, self.yellow_lower, self.yellow_upper)
        ratios = {
            'red': np.sum(red_mask > 0) / total,
            'green': np.sum(green_mask > 0) / total,
            'yellow': np.sum(yellow_mask > 0) / total,
        }
        best = max(ratios, key=ratios.get)
        if ratios[best] < self._color_threshold:
            return self._mean_hue_fallback(hsv_band)
        return best

    def _color_vote(self, roi: np.ndarray) -> str:
        """整 ROI 颜色投票（无竖长外形时的回退）。"""
        hsv = bgr_to_hsv_normalized(roi, use_clahe=self._use_clahe)
        total = roi.shape[0] * roi.shape[1]
        if total <= 0:
            return 'unknown'
        red_mask = (
            cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            | cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        )
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        ratios = {
            'red': np.sum(red_mask > 0) / total,
            'green': np.sum(green_mask > 0) / total,
            'yellow': np.sum(yellow_mask > 0) / total,
        }
        best = max(ratios, key=ratios.get)
        if ratios[best] < self._color_threshold:
            return self._mean_hue_fallback(hsv)
        return best

    def _mean_hue_fallback(self, hsv_roi: np.ndarray) -> str:
        """
        圆形均值色调回退：当所有颜色 mask 占比都不足阈值时（LED 过曝场景），
        用非暗非亮像素的圆形平均色调（circular mean）做分类。

        使用单位圆上的向量平均（atan2），正确处理红色跨越 0° 边界的情况。
        OpenCV H ∈ [0,179]，映射到 [0, 2π) 后计算圆形均值。
        """
        h = hsv_roi[:, :, 0].astype(np.float32)
        s = hsv_roi[:, :, 1].astype(np.float32)
        v = hsv_roi[:, :, 2].astype(np.float32)
        # 排除过暗（V<40）和过亮过曝低饱和（V>240 & S<15）的像素
        valid = (v > 40) & ~((v > 240) & (s < 15)) & (s > 8)
        if np.sum(valid) < 5:
            return 'unknown'
        hue_vals = h[valid]
        # OpenCV H ∈ [0, 179]，映射到弧度 [0, 2π)
        rad = hue_vals * (np.pi / 90.0)  # 乘 2 再乘 π/180 = π/90
        cos_sum = float(np.sum(np.cos(rad)))
        sin_sum = float(np.sum(np.sin(rad)))
        # 圆形均值角度（弧度），映射回 OpenCV H 空间
        mean_rad = np.arctan2(sin_sum, cos_sum)
        mean_h = (mean_rad * 90.0 / np.pi) % 180.0
        # 分类：红 ~0°/180°, 黄 ~30°, 绿 ~60°（OpenCV H 空间）
        if mean_h <= 18 or mean_h >= 160:
            return 'red'
        elif 18 < mean_h <= 48:
            return 'yellow'
        elif 48 < mean_h <= 95:
            return 'green'
        return 'unknown'

    def is_malfunctioning(self) -> bool:
        """检测信号灯是否故障（带滞回消抖，避免短暂遮挡误触发）。"""
        w = self._malfunction_window
        if len(self.state_history) < w:
            return False
        recent = list(self.state_history)[-w:]
        unknown_count = recent.count('unknown')
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        raw_fault = (unknown_count >= self._malfunction_unknown_thresh) or (
            changes >= self._malfunction_change_count
        )

        if self._malfunction_engaged:
            # 已锁存：需连续 5 帧正常才解除
            if raw_fault:
                self._malfunction_clear_counter = 0
            else:
                self._malfunction_clear_counter += 1
                if self._malfunction_clear_counter >= 5:
                    self._malfunction_engaged = False
                    self._malfunction_latch = 0
                    self._malfunction_clear_counter = 0
            return self._malfunction_engaged
        else:
            # 未锁存：需连续 3 帧故障才触发
            if raw_fault:
                self._malfunction_latch += 1
                if self._malfunction_latch >= 3:
                    self._malfunction_engaged = True
                    self._malfunction_clear_counter = 0
                    return True
            else:
                self._malfunction_latch = max(0, self._malfunction_latch - 1)
            return False

    def reset_history(self) -> None:
        """清空状态历史（摄像头移动 / 场景切换时调用）。"""
        self.state_history.clear()
        self._malfunction_latch = 0
        self._malfunction_engaged = False
        self._malfunction_clear_counter = 0
        self._red_frame_counter = 0


class StopLine:
    """停止线"""

    def __init__(self, y: int, x_start: int, x_end: int):
        self.y = y
        self.x_start = x_start
        self.x_end = x_end

    def is_crossed(self, prev_center: Tuple[int, int],
                   curr_center: Tuple[int, int]) -> bool:
        """判断是否越过停止线"""
        if not (self.x_start <= curr_center[0] <= self.x_end):
            return False
        if prev_center[1] < self.y <= curr_center[1]:
            return True
        return False


class AdaptiveViolationDetector:
    """
    自适应违规检测器

    创新点：
    1. 检测特种车辆和交警
    2. 违规时如果附近有特种车辆或交警，标记为"异常"
    3. 异常情况单独保存，便于人工复核
    """

    def __init__(self,
                 speed_limit: float = 60.0,
                 stop_line: Optional[StopLine] = None,
                 snapshot_dir: str = "data/snapshots",
                 emergency_distance: int = DEFAULT_EMERGENCY_DISTANCE_PX,
                 red_light_enabled: bool = True,
                 speeding_enabled: bool = True,
                 wrong_way_enabled: bool = True,
                 illegal_lane_enabled: bool = True,
                 expected_flow_direction: str = "south",
                 lane_change_lateral_px: int = DEFAULT_LANE_CHANGE_LATERAL_PX,
                 lane_change_min_speed_kmh: float = DEFAULT_LANE_CHANGE_MIN_SPEED_KMH,
                 traffic_light_config: Optional[Dict] = None,
                 emergency_vehicle_config: Optional[Dict] = None,
                 traffic_police_config: Optional[Dict] = None):
        self.speed_limit = speed_limit
        self.stop_line = stop_line
        self.snapshot_dir = Path(snapshot_dir)
        self.emergency_distance = emergency_distance
        self.red_light_enabled = red_light_enabled
        self.speeding_enabled = speeding_enabled
        self.wrong_way_enabled = wrong_way_enabled
        self.illegal_lane_enabled = illegal_lane_enabled
        self.expected_flow_direction = expected_flow_direction.lower()
        self.lane_change_lateral_px = lane_change_lateral_px
        self.lane_change_min_speed_kmh = lane_change_min_speed_kmh
        self.track_history_maxlen = DEFAULT_LANE_CHANGE_HISTORY_LEN
        self.lane_analyzer = build_analyzer_from_violation_config(
            expected_flow_direction=self.expected_flow_direction,
            lane_change_lateral_px=lane_change_lateral_px,
            lane_change_min_speed_kmh=lane_change_min_speed_kmh,
            history_len=self.track_history_maxlen,
        )

        # 创建截图目录
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        (self.snapshot_dir / "violations").mkdir(exist_ok=True)
        (self.snapshot_dir / "anomaly").mkdir(exist_ok=True)

        # 子模块
        self.traffic_light_detector = TrafficLightDetector(traffic_light_config)
        self.emergency_detector = EmergencyVehicleDetector(emergency_vehicle_config)
        self.police_detector = TrafficPoliceDetector(traffic_police_config)

        # 状态
        self.track_history: Dict[int, deque] = {}
        self.recorded_violations: Dict[int, Dict] = {}
        self.current_light_state = 'unknown'
        self.current_emergency_vehicles: List[EmergencyVehicle] = []
        self.current_police_bboxes: List[Tuple[int, int, int, int]] = []

        # 统计
        self.total_violations = 0
        self.anomaly_count = 0

        # 停止线动态重检测
        tl_cfg = traffic_light_config or {}
        self._stop_line_redetect_interval = int(tl_cfg.get('stop_line_redetect_interval', 0))
        self._stop_line_auto_detect = bool(tl_cfg.get('stop_line_auto_detect', False))
        self._frame_count = 0

    def set_stop_line(self, y: int, x_start: int, x_end: int):
        """设置停止线"""
        self.stop_line = StopLine(y, x_start, x_end)

    def update(self, frame: np.ndarray,
               vehicle_bboxes: List[Tuple[int, int, int, int]],
               person_bboxes: List[Tuple[int, int, int, int]] = None,
               light_bbox: Optional[Tuple[int, int, int, int]] = None):
        """更新检测器状态"""
        self._frame_count += 1

        if light_bbox:
            self.current_light_state = self.traffic_light_detector.detect_state(
                frame, light_bbox
            )

        # 检测特种车辆
        self.current_emergency_vehicles = self.emergency_detector.detect(
            frame, vehicle_bboxes
        )

        # 检测交警
        if person_bboxes:
            self.current_police_bboxes = self.police_detector.detect(
                frame, person_bboxes
            )
        else:
            self.current_police_bboxes = []

        # 停止线动态重检测（适应相机微小移动）
        if (self._stop_line_auto_detect
            and self._stop_line_redetect_interval > 0
            and self._frame_count % self._stop_line_redetect_interval == 0):
            from src.utils.stop_line_detect import detect_stop_line
            result = detect_stop_line(frame)
            if result is not None:
                y, x_start, x_end = result
                self.stop_line = StopLine(y, x_start, x_end)
                logger.debug(
                    "Stop line re-detected at y=%d x=[%d, %d] (frame %d)",
                    y, x_start, x_end, self._frame_count,
                )

    def check_violation(self,
                        track_id: int,
                        bbox: Tuple[int, int, int, int],
                        speed: float,
                        frame: np.ndarray,
                        plate_number: Optional[str] = None,
                        direction: Optional[Direction] = None) -> Optional[ViolationRecord]:
        """检查单个车辆的违规行为"""
        fh, fw = frame.shape[:2]
        self.lane_analyzer.set_frame_scale(fh)
        clamped = clamp_bbox(bbox, fw, fh)
        if clamped is None:
            return None
        x1, y1, x2, y2 = clamped
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.track_history_maxlen)
            self.recorded_violations[track_id] = {}

        history = self.track_history[track_id]

        violation_type = None
        violation_speed = None

        # 优先级：闯红灯 > 超速 > 逆行 > 违规变道
        # 使用平滑后的灯态，并确认红灯在车辆越线前已亮起足够帧数
        smoothed_state = self.traffic_light_detector.get_smoothed_state()
        if (self.red_light_enabled and
            self.stop_line and
            len(history) > 0 and
            smoothed_state == 'red' and
            self.traffic_light_detector.red_duration_frames() >= self.traffic_light_detector._min_red_frames):
            prev_center = history[-1]
            if self.stop_line.is_crossed(prev_center, center):
                if self._can_record(track_id, ViolationType.RED_LIGHT):
                    violation_type = ViolationType.RED_LIGHT

        if violation_type is None and self.speeding_enabled and speed > self.speed_limit:
            if self._can_record(track_id, ViolationType.SPEEDING):
                violation_type = ViolationType.SPEEDING
                violation_speed = speed

        trajectory = list(history) + [center]
        lane_result = self.lane_analyzer.analyze(
            trajectory,
            speed_kmh=speed,
            direction=direction,
            wrong_way_enabled=self.wrong_way_enabled,
            illegal_lane_enabled=self.illegal_lane_enabled,
        )

        if (
            violation_type is None
            and lane_result.is_wrong_way
            and self._can_record(track_id, ViolationType.WRONG_WAY)
        ):
            violation_type = ViolationType.WRONG_WAY
            violation_speed = speed

        if (
            violation_type is None
            and lane_result.is_illegal_lane_change
            and self._can_record(track_id, ViolationType.ILLEGAL_LANE)
        ):
            violation_type = ViolationType.ILLEGAL_LANE
            violation_speed = speed

        history.append(center)

        if violation_type is None:
            return None

        # 检查是否为异常情况
        is_anomaly, anomaly_reason, nearby_objects = self._check_anomaly(bbox)

        # 生成记录
        timestamp = datetime.now()
        record_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        # 保存截图
        snapshot_path = self._save_snapshot(
            frame, record_id, is_anomaly, bbox, violation_type
        )

        record = ViolationRecord(
            record_id=record_id,
            violation_type=violation_type,
            track_id=track_id,
            timestamp=timestamp,
            location=center,
            speed=violation_speed,
            plate_number=plate_number,
            snapshot_path=snapshot_path,
            is_anomaly=is_anomaly,
            anomaly_reason=anomaly_reason,
            nearby_objects=nearby_objects
        )

        # 更新统计
        self.total_violations += 1
        if is_anomaly:
            self.anomaly_count += 1

        self.recorded_violations[track_id][violation_type] = time.time()
        return record

    def _can_record(self, track_id: int, vtype: ViolationType) -> bool:
        """检查是否可记录该违规（冷却时间未过则跳过）。"""
        last = self.recorded_violations.get(track_id, {}).get(vtype)
        if last is None:
            return True
        return (time.time() - last) >= VIOLATION_COOLDOWN_SEC

    def _check_anomaly(self, vehicle_bbox: Tuple[int, int, int, int]
                       ) -> Tuple[bool, AnomalyReason, List[str]]:
        """
        检查是否为异常情况（附近有特种车辆或交警）

        Returns:
            (是否异常, 异常原因, 附近特殊对象列表)
        """
        nearby_objects = []
        vx = (vehicle_bbox[0] + vehicle_bbox[2]) // 2
        vy = (vehicle_bbox[1] + vehicle_bbox[3]) // 2

        # 1. 检查附近是否有特种车辆
        for ev in self.current_emergency_vehicles:
            ex = (ev.bbox[0] + ev.bbox[2]) // 2
            ey = (ev.bbox[1] + ev.bbox[3]) // 2
            distance = np.sqrt((vx - ex)**2 + (vy - ey)**2)

            if distance < self.emergency_distance:
                ev_name = self._get_emergency_vehicle_name(ev.vehicle_type)
                nearby_objects.append(ev_name)

        if nearby_objects:
            return True, AnomalyReason.EMERGENCY_VEHICLE, nearby_objects

        # 2. 检查附近是否有交警
        for police_bbox in self.current_police_bboxes:
            px = (police_bbox[0] + police_bbox[2]) // 2
            py = (police_bbox[1] + police_bbox[3]) // 2
            distance = np.sqrt((vx - px)**2 + (vy - py)**2)

            if distance < self.emergency_distance:
                nearby_objects.append("交警")

        if nearby_objects:
            return True, AnomalyReason.TRAFFIC_POLICE, nearby_objects

        # 3. 检查信号灯故障
        if self.traffic_light_detector.is_malfunctioning():
            return True, AnomalyReason.SIGNAL_MALFUNCTION, ["信号灯故障"]

        return False, AnomalyReason.NONE, []

    def _get_emergency_vehicle_name(self, ev_type: EmergencyVehicleType) -> str:
        """获取特种车辆中文名称"""
        names = {
            EmergencyVehicleType.AMBULANCE: "救护车",
            EmergencyVehicleType.FIRE_TRUCK: "消防车",
            EmergencyVehicleType.POLICE_CAR: "警车",
            EmergencyVehicleType.RESCUE_VEHICLE: "工程救险车",
            EmergencyVehicleType.UNKNOWN: "特种车辆",
        }
        return names.get(ev_type, "特种车辆")

    def _save_snapshot(self, frame: np.ndarray, record_id: str,
                       is_anomaly: bool, bbox: Tuple[int, int, int, int],
                       violation_type: ViolationType) -> str:
        """保存截图"""
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox

        # 黄色=异常待复核，红色=正常违规
        color = (0, 255, 255) if is_anomaly else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{violation_type.value}"
        if is_anomaly:
            label += " [ANOMALY]"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, time_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        subdir = "anomaly" if is_anomaly else "violations"
        filename = f"{record_id}_{violation_type.value}.jpg"
        filepath = self.snapshot_dir / subdir / filename

        try:
            cv2.imwrite(str(filepath), annotated)
        except cv2.error as exc:
            logger.error("Failed to save snapshot %s: %s", filepath, exc)
            return ""
        return str(filepath)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_violations": self.total_violations,
            "anomaly_count": self.anomaly_count,
            "normal_violations": self.total_violations - self.anomaly_count,
            # 兼容旧代码
            "exempted_count": self.anomaly_count,
            "actual_violations": self.total_violations - self.anomaly_count,
            "exemption_rate": self.anomaly_count / max(1, self.total_violations),
        }

    def clear_track(self, track_id: int):
        """清除跟踪记录"""
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.recorded_violations:
            del self.recorded_violations[track_id]

    def cleanup_stale_tracks(self, active_ids: set) -> int:
        """清理已消失的 track 历史，返回清理数量。"""
        stale = [tid for tid in self.track_history if tid not in active_ids]
        for tid in stale:
            self.clear_track(tid)
        return len(stale)

    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制标注"""
        annotated = frame.copy()

        # 绘制停止线
        if self.stop_line:
            if self.current_light_state == 'red':
                color = (0, 0, 255)      # 红灯 → 红线
            elif self.current_light_state == 'green':
                color = (0, 255, 0)      # 绿灯 → 绿线
            elif self.current_light_state == 'yellow':
                color = (0, 255, 255)    # 黄灯 → 黄线
            else:
                color = (0, 200, 255)    # 未知 → 橙线（非绿色，避免误导）
            cv2.line(annotated,
                     (self.stop_line.x_start, self.stop_line.y),
                     (self.stop_line.x_end, self.stop_line.y),
                     color, 2)

        # 绘制交通灯状态（圆点 + 文字标签）
        light_colors = {'red': (0, 0, 255), 'green': (0, 255, 0),
                        'yellow': (0, 255, 255), 'unknown': (0, 140, 255)}
        cv2.circle(annotated, (30, 60), 15,
                   light_colors.get(self.current_light_state, (128, 128, 128)), -1)
        state_text = self.current_light_state.upper() if self.current_light_state != 'unknown' else '?'
        cv2.putText(annotated, f"TL:{state_text}", (55, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 标注特种车辆
        for ev in self.current_emergency_vehicles:
            x1, y1, x2, y2 = ev.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # OpenCV 默认字体不支持中文，这里用英文缩写避免出现“????”
            ev_labels_en = {
                EmergencyVehicleType.AMBULANCE: "Ambulance",
                EmergencyVehicleType.FIRE_TRUCK: "Fire Truck",
                EmergencyVehicleType.POLICE_CAR: "Police Car",
                EmergencyVehicleType.RESCUE_VEHICLE: "Rescue",
                EmergencyVehicleType.UNKNOWN: "Emergency",
            }
            label_en = ev_labels_en.get(ev.vehicle_type, "Emergency")
            cv2.putText(annotated, label_en, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 标注交警
        for bbox in self.current_police_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 3)
            cv2.putText(annotated, "Traffic Police", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

        # 显示统计
        stats = self.get_statistics()
        info = f"Violations: {stats['normal_violations']} | Anomaly: {stats['anomaly_count']}"
        cv2.putText(annotated, info, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated
