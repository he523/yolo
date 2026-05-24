"""YOLO 目标检测模块"""
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from ultralytics import YOLO

from src.utils.bbox import clamp_bbox, iou_xyxy_batch
from src.utils.constants import (
    DEFAULT_TILING_MIN_DETS,
    DEFAULT_TILING_OVERLAP,
    DEFAULT_TILING_INTERVAL_FRAMES,
)
from src.utils.model_paths import resolve_yolo_model

logger = logging.getLogger(__name__)



@dataclass
class Detection:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[int, int]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        """获取边界框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_tlwh(self) -> Tuple[int, int, int, int]:
        """转换为 (top, left, width, height) 格式"""
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2 - x1, y2 - y1


class VehicleDetector:
    """车辆检测器"""

    # COCO 数据集中的车辆类别
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    # 交通灯类别
    TRAFFIC_LIGHT_CLASS = 9
    # 人员类别（用于交警检测输入）
    PERSON_CLASS = 0

    def __init__(self,
                 model_path: str = "yolo12n.pt",
                 confidence: float = 0.2,
                 iou_threshold: float = 0.45,
                 device: str = "cuda",
                 imgsz: int = 960,
                 max_det: int = 300,
                 enable_tiling: bool = True,
                 tiling_grid: Tuple[int, int] = (2, 2),
                 tiling_overlap: float = 0.20,
                 tiling_min_dets: int = DEFAULT_TILING_MIN_DETS,
                 tiling_interval: int = DEFAULT_TILING_INTERVAL_FRAMES,
                 tiling_mode: str = "strip",
                 vehicle_classes: Optional[Dict[int, str]] = None):
        """
        初始化检测器

        Args:
            model_path: YOLO 模型路径
            confidence: 置信度阈值
            iou_threshold: NMS IOU 阈值
            device: 运行设备 (cuda/cpu)
            imgsz: 推理输入尺寸（越大越利于远处小目标，但更慢）
            max_det: 最大检测数量上限
            enable_tiling: 是否启用切片检测兜底（提升远处小目标召回）
            tiling_grid: 切片网格 (rows, cols)
            tiling_overlap: 切片重叠比例 (0~0.5)
            tiling_min_dets: 常规检测结果少于该值时触发切片兜底
            tiling_interval: 切片兜底最少间隔帧数（降低开销）
            tiling_mode: strip=仅上半幅2块(快) / full=完整网格(慢但更全)
            vehicle_classes: 自定义车辆类别映射 {class_id: name}；None 使用默认 COCO 映射
        """
        resolved, fallback = resolve_yolo_model(model_path)
        if fallback:
            logger.warning("Vehicle detector using fallback weights; fine-tune for traffic classes")
        self.model = YOLO(resolved)
        # 允许通过 settings.yaml 的 detector.classes 自定义车辆类别
        # 支持 list[int]（从模型 names 自动查名称）或 dict[int, str]
        if vehicle_classes is not None:
            if isinstance(vehicle_classes, dict):
                self.VEHICLE_CLASSES = dict(vehicle_classes)
            else:
                self.VEHICLE_CLASSES = {
                    int(cid): self.model.names.get(int(cid), f'class_{cid}')
                    for cid in vehicle_classes
                }
        self._frame_index = 0
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = self.model.names
        self.imgsz = int(imgsz) if imgsz else 960
        self.max_det = int(max_det) if max_det else 300
        self.enable_tiling = bool(enable_tiling)
        self.tiling_grid = tiling_grid
        self.tiling_overlap = float(tiling_overlap or DEFAULT_TILING_OVERLAP)
        self.tiling_min_dets = int(tiling_min_dets)
        self.tiling_interval = max(1, int(tiling_interval))
        self.tiling_mode = (tiling_mode or "strip").lower()

    def detect(self, frame: np.ndarray,
               classes: Optional[List[int]] = None) -> List[Detection]:
        """
        检测图像中的目标

        Args:
            frame: BGR 格式的图像帧
            classes: 要检测的类别ID列表，None 表示检测所有车辆类别

        Returns:
            检测结果列表
        """
        if classes is None:
            classes = list(self.VEHICLE_CLASSES.keys())

        self._frame_index += 1
        detections = self._predict_frame(frame, classes)
        # 自适应切片兜底：降频 + 默认仅上半幅条带（比 2x2 全图快约 2–4 倍）
        run_tiling = (
            self.enable_tiling
            and len(detections) < self.tiling_min_dets
            and (self._frame_index % self.tiling_interval == 0)
        )
        if run_tiling:
            tiled = self._predict_tiled(frame, classes)
            if tiled:
                detections = self._merge_detections(detections + tiled, iou_thr=0.55)

        return detections

    def _predict_frame(self, frame: np.ndarray, classes: List[int]) -> List[Detection]:
        """对整帧做一次常规 YOLO 推理。"""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=classes,
            device=self.device,
            imgsz=self.imgsz,
            max_det=self.max_det,
            half=(self.device == "cuda"),
            verbose=False,
        )
        fh, fw = frame.shape[:2]
        return self._results_to_detections(results, frame_w=fw, frame_h=fh)

    def _predict_tiled(self, frame: np.ndarray, classes: List[int]) -> List[Detection]:
        """切片推理；strip 模式仅处理画面上半区（远处小目标常见区域）。"""
        h, w = frame.shape[:2]
        if self.tiling_mode == "strip":
            return self._predict_strip_tiles(frame, classes, h, w)

        rows, cols = self.tiling_grid
        rows = max(1, int(rows))
        cols = max(1, int(cols))
        ov = min(max(self.tiling_overlap, 0.0), 0.45)

        tile_w = max(1, int(w / cols))
        tile_h = max(1, int(h / rows))
        step_x = max(1, int(tile_w * (1 - ov)))
        step_y = max(1, int(tile_h * (1 - ov)))

        all_dets: List[Detection] = []
        y = 0
        while y < h:
            x = 0
            y2 = min(h, y + tile_h)
            y1 = max(0, y2 - tile_h)
            while x < w:
                x2 = min(w, x + tile_w)
                x1 = max(0, x2 - tile_w)
                tile = frame[y1:y2, x1:x2]
                results = self.model.predict(
                    tile,
                    conf=max(0.05, self.confidence * 0.85),  # 切片略降阈值，提高召回
                    iou=self.iou_threshold,
                    classes=classes,
                    device=self.device,
                    imgsz=self.imgsz,
                    max_det=self.max_det,
                    half=(self.device == "cuda"),
                    verbose=False,
                )
                th, tw = tile.shape[:2]
                dets = self._results_to_detections(results, frame_w=tw, frame_h=th)
                for d in dets:
                    bx1, by1, bx2, by2 = d.bbox
                    mapped = clamp_bbox(
                        (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1), w, h,
                    )
                    if mapped is None:
                        continue
                    bx1, by1, bx2, by2 = mapped
                    all_dets.append(Detection(
                        bbox=(bx1, by1, bx2, by2),
                        confidence=d.confidence,
                        class_id=d.class_id,
                        class_name=d.class_name,
                    ))
                x += step_x
                if x >= w:
                    break
            y += step_y
            if y >= h:
                break

        return self._merge_detections(all_dets, iou_thr=0.55)

    def _predict_strip_tiles(self, frame: np.ndarray, classes: List[int],
                             h: int, w: int) -> List[Detection]:
        """上半幅左右两块，兼顾远处小车与速度。"""
        y1, y2 = 0, max(1, h // 2)
        mid = w // 2
        regions = [(0, y1, mid, y2), (mid, y1, w, y2)]
        all_dets: List[Detection] = []
        for rx1, ry1, rx2, ry2 in regions:
            tile = frame[ry1:ry2, rx1:rx2]
            if tile.size == 0:
                continue
            results = self.model.predict(
                tile,
                conf=max(0.05, self.confidence * 0.85),
                iou=self.iou_threshold,
                classes=classes,
                device=self.device,
                imgsz=self.imgsz,
                max_det=self.max_det,
                half=(self.device == "cuda"),
                verbose=False,
            )
            for d in self._results_to_detections(results, frame_w=w, frame_h=h):
                bx1, by1, bx2, by2 = d.bbox
                all_dets.append(Detection(
                    bbox=(bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1),
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                ))
        return self._merge_detections(all_dets, iou_thr=0.55)

    def _results_to_detections(self, results,
                               frame_w: Optional[int] = None,
                               frame_h: Optional[int] = None) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            if frame_w is None and getattr(result, 'orig_shape', None):
                frame_h, frame_w = result.orig_shape[:2]
            for box in boxes:
                raw = box.xyxy[0].tolist()
                if frame_w and frame_h:
                    clamped = clamp_bbox(
                        tuple(int(round(v)) for v in raw),
                        frame_w, frame_h,
                    )
                    if clamped is None:
                        continue
                    x1, y1, x2, y2 = clamped
                else:
                    x1, y1, x2, y2 = map(int, raw)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, 'unknown')
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))
        return detections

    def _merge_detections(self, dets: List[Detection], iou_thr: float = 0.55) -> List[Detection]:
        """按类别做简单 NMS 合并，减少重复框。"""
        if not dets:
            return []

        by_cls: Dict[int, List[Detection]] = {}
        for d in dets:
            by_cls.setdefault(d.class_id, []).append(d)

        merged: List[Detection] = []
        for cls_id, items in by_cls.items():
            items = sorted(items, key=lambda d: d.confidence, reverse=True)
            kept: List[Detection] = []
            kept_box_list: List[List[float]] = []  # 收集列表，最后批量构建 np 数组
            for d in items:
                if kept_box_list:
                    overlaps = iou_xyxy_batch(d.bbox, np.array(kept_box_list, dtype=np.float64)) >= iou_thr
                    if np.any(overlaps):
                        continue
                kept.append(d)
                kept_box_list.append(list(d.bbox))
            merged.extend(kept)
        return merged

    def detect_vehicles(self, frame: np.ndarray) -> List[Detection]:
        """仅检测车辆"""
        return self.detect(frame, list(self.VEHICLE_CLASSES.keys()))

    # detect_traffic_lights / detect_persons / detect_all 已删除（未被使用）
    # 上下文检测由 VideoThread / main.py 直接调用 detect() 并自行分类
