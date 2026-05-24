"""
PyQt5 Main Window - Real-Time Traffic Analysis System
Supports adaptive violation detection, ST-GAT interaction modeling,
and collision risk prediction.
"""
import cv2
import numpy as np
import os
import re
import time
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTableWidget,
    QTableWidgetItem, QTabWidget, QGroupBox, QLineEdit,
    QComboBox, QSpinBox, QStatusBar, QSplitter, QMessageBox,
    QCheckBox, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from typing import Optional, List, Dict
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.utils.matplotlib_zh import configure_matplotlib_chinese, chart_font_props

configure_matplotlib_chinese()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.video import VideoStream
from src.core import VehicleDetector, ByteTracker, FeatureExtractor
from src.core.adaptive_violation import AdaptiveViolationDetector, ViolationRecord, AnomalyReason
from src.core.lane_violation import build_analyzer_from_violation_config
from src.core.stgat import VehicleInteractionGraph
from src.core.collision_risk import CollisionRiskPredictor, RiskLevel
from src.ocr import PlateReader
from src.ocr.ocr_scheduler import OCRScheduler
from src.database import Database
from src.database.scheduler import start_db_cleanup_from_config
from src.utils.config import load_config, save_config_section
from src.utils.model_manager import ModelManager
from src.utils.performance import PerformanceOptimizer, FPSMonitor
from src.utils.stop_line_detect import detect_stop_line
from src.gui.i18n import Translator, SUPPORTED_LANGUAGES
from src.gui.theme import (
    COLORS,
    MetricCard,
    build_app_header,
    global_stylesheet,
    insight_panel_style,
    message_box_stylesheet,
)


class StatisticsCanvas(FigureCanvas):
    """Matplotlib canvas for statistics charts"""

    def __init__(self, parent=None, translator: Optional[Translator] = None):
        self._tr = translator
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=COLORS['chart_bg'])
        super().__init__(self.fig)
        self.setParent(parent)

        # Data storage
        self.time_data = deque(maxlen=60)
        self.vehicle_count_data = deque(maxlen=60)
        self.speed_data = deque(maxlen=100)
        self.violation_counts: Dict[str, int] = {}
        self.vehicle_type_counts: Dict[str, int] = {}

        # Create subplots
        self.ax_flow = self.fig.add_subplot(2, 2, 1)
        self.ax_violation = self.fig.add_subplot(2, 2, 2)
        self.ax_speed = self.fig.add_subplot(2, 2, 3)
        self.ax_type = self.fig.add_subplot(2, 2, 4)

        self._setup_style()
        self.fig.tight_layout(pad=2.0)

    def set_translator(self, translator: Translator) -> None:
        self._tr = translator
        self._setup_style()
        self._redraw()

    def _chart_title(self, key: str) -> str:
        if self._tr:
            return self._tr.tr(key)
        return key

    def _setup_style(self):
        """Setup chart style"""
        for ax in [self.ax_flow, self.ax_violation, self.ax_speed, self.ax_type]:
            ax.set_facecolor(COLORS['surface'])
            ax.tick_params(colors=COLORS['text_muted'], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLORS['chart_grid'])
            ax.title.set_color(COLORS['text'])

        self.ax_flow.set_title(self._chart_title('chart_flow'), fontsize=10, color=COLORS['text'])
        self.ax_violation.set_title(self._chart_title('chart_violation'), fontsize=10, color=COLORS['text'])
        self.ax_speed.set_title(self._chart_title('chart_speed'), fontsize=10, color=COLORS['text'])
        self.ax_type.set_title(self._chart_title('chart_type'), fontsize=10, color=COLORS['text'])

    def update_data(self, vehicle_count: int, speeds: List[float],
                    violations: Dict[str, int], vehicle_types: Dict[str, int]):
        """Update chart data"""
        self.time_data.append(datetime.now().strftime('%H:%M:%S'))
        self.vehicle_count_data.append(vehicle_count)

        for speed in speeds:
            if speed > 0:
                self.speed_data.append(speed)

        for vtype, count in violations.items():
            self.violation_counts[vtype] = self.violation_counts.get(vtype, 0) + count

        for vtype, count in vehicle_types.items():
            self.vehicle_type_counts[vtype] = self.vehicle_type_counts.get(vtype, 0) + count

        self._redraw()

    def _redraw(self):
        """Redraw all charts"""
        # Traffic flow
        self.ax_flow.clear()
        self._setup_ax(self.ax_flow, self._chart_title('chart_flow'))
        if self.vehicle_count_data:
            x = list(range(len(self.vehicle_count_data)))
            self.ax_flow.plot(x, list(self.vehicle_count_data), color=COLORS['text'], linewidth=2)
            self.ax_flow.fill_between(x, list(self.vehicle_count_data), alpha=0.15, color=COLORS['text'])
            self.ax_flow.set_ylabel(self._chart_title('chart_y_vehicles'), fontsize=8, color=COLORS['text_muted'])

        # Violation pie
        self.ax_violation.clear()
        self._setup_ax(self.ax_violation, self._chart_title('chart_violation'))
        if self.violation_counts:
            labels = list(self.violation_counts.keys())
            sizes = list(self.violation_counts.values())
            greys = ['#ffffff', '#d4d4d4', '#a3a3a3', '#737373'][:len(labels)]
            self.ax_violation.pie(
                sizes,
                labels=labels,
                colors=greys,
                autopct='%1.0f%%',
                textprops=chart_font_props(color='white', fontsize=7),
            )
        else:
            nodata = self._chart_title('chart_no_data')
            self.ax_violation.text(0.5, 0.5, nodata, ha='center', va='center', color=COLORS['text_muted'], fontsize=10)

        # Speed histogram
        self.ax_speed.clear()
        self._setup_ax(self.ax_speed, self._chart_title('chart_speed'))
        if self.speed_data:
            self.ax_speed.hist(list(self.speed_data), bins=15, color='#d4d4d4', alpha=0.85, edgecolor=COLORS['chart_grid'])
            self.ax_speed.axvline(x=60, color=COLORS['text'], linestyle='--', linewidth=1.5)
            self.ax_speed.set_xlabel('km/h', fontsize=8, color=COLORS['text_muted'])
        else:
            nodata = self._chart_title('chart_no_data')
            self.ax_speed.text(0.5, 0.5, nodata, ha='center', va='center', color=COLORS['text_muted'], fontsize=10)

        # Vehicle type bar
        self.ax_type.clear()
        self._setup_ax(self.ax_type, self._chart_title('chart_type'))
        if self.vehicle_type_counts:
            types = list(self.vehicle_type_counts.keys())
            counts = list(self.vehicle_type_counts.values())
            greys = ['#ffffff', '#d4d4d4', '#a3a3a3', '#737373'][:len(types)]
            self.ax_type.bar(types, counts, color=greys)
            self.ax_type.tick_params(axis='x', labelrotation=15)
        else:
            nodata = self._chart_title('chart_no_data')
            self.ax_type.text(0.5, 0.5, nodata, ha='center', va='center', color=COLORS['text_muted'], fontsize=10)

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def _setup_ax(self, ax, title: str):
        """Setup axis style"""
        ax.set_facecolor(COLORS['surface'])
        ax.tick_params(colors=COLORS['text_muted'], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(COLORS['chart_grid'])
        ax.set_title(title, fontsize=10, color=COLORS['text'])

    def reset(self):
        """Reset all data"""
        self.time_data.clear()
        self.vehicle_count_data.clear()
        self.speed_data.clear()
        self.violation_counts.clear()
        self.vehicle_type_counts.clear()
        self._redraw()


class VideoThread(QThread):
    """Video processing thread"""
    frame_ready = pyqtSignal(np.ndarray, list, list, list, list, dict)  # Added plate_results
    stats_updated = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, source: str, config: dict):
        super().__init__()
        self.source = source
        self.config = config
        self.running = False
        self._paused = False

        self.video_stream: Optional[VideoStream] = None
        self.detector: Optional[VehicleDetector] = None
        self.tracker: Optional[ByteTracker] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.violation_detector: Optional[AdaptiveViolationDetector] = None
        self.interaction_graph: Optional[VehicleInteractionGraph] = None
        self.collision_predictor: Optional[CollisionRiskPredictor] = None
        self.plate_reader: Optional[PlateReader] = None
        self._perf_optimizer: Optional[PerformanceOptimizer] = None
        self._fps_monitor: Optional[FPSMonitor] = None
        self._runtime_perf_enabled = False

    def apply_runtime_settings(self, settings: dict) -> None:
        """Apply GUI settings while video processing is running."""
        if not settings:
            return

        self.config.update(settings)

        if self.detector is not None:
            if 'confidence' in settings:
                self.detector.confidence = float(settings['confidence'])
            enable_tiling = bool(self.config.get('enable_tiling', False))
            if self._perf_optimizer is not None:
                self._perf_optimizer.base_enable_tiling = enable_tiling
                if not enable_tiling:
                    self._perf_optimizer._plan.enable_tiling = False
            if not self._runtime_perf_enabled:
                self.detector.enable_tiling = enable_tiling
                self.detector.imgsz = int(self.config.get('imgsz', 768))

        if 'performance_enabled' in settings:
            self._runtime_perf_enabled = bool(settings['performance_enabled'])
            if (
                self._runtime_perf_enabled
                and self._fps_monitor is None
                and self._perf_optimizer is not None
            ):
                self._fps_monitor = FPSMonitor(
                    target_fps=float(
                        self.config.get('performance_target_fps', self.config.get('fps', 15))
                    ),
                    warmup_frames=int(self.config.get('performance_warmup_frames', 45)),
                    low_fps_checks=int(self.config.get('performance_low_fps_checks', 5)),
                )
                self._fps_monitor.mark_ready()
            if not self._runtime_perf_enabled and self.detector is not None:
                self.detector.imgsz = int(self.config.get('imgsz', 768))
                self.detector.enable_tiling = bool(self.config.get('enable_tiling', False))

        if self.violation_detector is None:
            return

        if 'wrong_way_enabled' in settings:
            self.violation_detector.wrong_way_enabled = bool(settings['wrong_way_enabled'])
        if 'illegal_lane_enabled' in settings:
            self.violation_detector.illegal_lane_enabled = bool(settings['illegal_lane_enabled'])
        if 'speed_limit' in settings:
            self.violation_detector.speed_limit = float(settings['speed_limit'])
        if 'emergency_distance' in settings:
            self.violation_detector.emergency_distance = int(settings['emergency_distance'])
        if 'expected_flow_direction' in settings:
            flow = str(settings['expected_flow_direction']).lower()
            if flow != self.violation_detector.expected_flow_direction:
                self.violation_detector.expected_flow_direction = flow
                self.violation_detector.lane_analyzer = build_analyzer_from_violation_config(
                    expected_flow_direction=flow,
                    lane_change_lateral_px=self.violation_detector.lane_change_lateral_px,
                    lane_change_min_speed_kmh=self.violation_detector.lane_change_min_speed_kmh,
                    history_len=self.violation_detector.track_history_maxlen,
                )
        if 'stop_line' in settings:
            stop_line = settings['stop_line']
            if stop_line:
                self.violation_detector.set_stop_line(
                    int(stop_line['y']),
                    int(stop_line['x_start']),
                    int(stop_line['x_end']),
                )
            else:
                self.violation_detector.stop_line = None

    def run(self):
        """Run video processing"""
        try:
            self.video_stream = VideoStream(
                self.source,
                fps=self.config.get('fps', 15),
                resize=tuple(self.config['resize']) if self.config.get('resize') else None,
            )
            if not self.video_stream.open():
                lang = self.config.get("gui_language", "zh_CN")
                tr = Translator(lang)
                self.error.emit(tr.tr("error_open_source"))
                return

            model_mgr = ModelManager({'detector': {'model_path': self.config.get('model_path')},
                                      'ocr': {'model_path': self.config.get('plate_model_path')}})
            yolo_path, _ = model_mgr.load_yolo_path(self.config.get('model_path'))

            perf_cfg = {
                'enabled': self.config.get('performance_enabled', True),
                'target_fps': self.config.get('performance_target_fps', self.config.get('fps', 15)),
                'dynamic_resolution': self.config.get('performance_dynamic_resolution', True),
                'frame_skip': self.config.get('performance_frame_skip', 1),
                'enable_tiling': self.config.get('enable_tiling', False),
            }
            perf_enabled = bool(perf_cfg.get('enabled', True))
            perf = PerformanceOptimizer.from_config(perf_cfg, base_imgsz=self.config.get('imgsz', 768))
            perf.base_enable_tiling = bool(self.config.get('enable_tiling', False))
            perf._plan.enable_tiling = perf.base_enable_tiling
            fps_monitor = (
                FPSMonitor(
                    target_fps=float(perf_cfg.get('target_fps', 15)),
                    warmup_frames=int(self.config.get('performance_warmup_frames', 45)),
                    low_fps_checks=int(self.config.get('performance_low_fps_checks', 5)),
                )
                if perf_enabled else None
            )
            self._perf_optimizer = perf
            self._fps_monitor = fps_monitor
            self._runtime_perf_enabled = perf_enabled

            self.detector = VehicleDetector(
                model_path=yolo_path,
                confidence=self.config.get('confidence', 0.3),
                iou_threshold=self.config.get('iou_threshold', 0.45),
                device=self.config.get('device', 'cpu'),
                imgsz=self.config.get('imgsz', 768),
                max_det=self.config.get('max_det', 300),
                enable_tiling=self.config.get('enable_tiling', False),
                tiling_grid=tuple(self.config.get('tiling_grid', (2, 2))),
                tiling_overlap=self.config.get('tiling_overlap', 0.20),
                tiling_min_dets=self.config.get('tiling_min_dets', 10),
                tiling_interval=self.config.get('tiling_interval', 5),
                tiling_mode=self.config.get('tiling_mode', 'strip'),
                vehicle_classes=self.config.get('vehicle_classes'),
            )
            # 如果车辆检测模型不包含 traffic light 类，尝试使用 COCO 模型做上下文检测
            self.context_detector = None
            if not self.detector.has_traffic_light_class():
                import logging as _logging
                _log = _logging.getLogger(__name__)
                # 按优先级查找本地已有的 COCO 模型（不触发下载，避免网络超时）
                _coco_candidates = ['models/yolo12n.pt', 'models/yolo11n.pt', 'models/yolov8n.pt']
                _coco_path = None
                for _c in _coco_candidates:
                    if Path(_c).exists():
                        _coco_path = _c
                        break
                if _coco_path:
                    _log.info("Primary model lacks traffic-light class; using %s for context detection", _coco_path)
                    self.context_detector = VehicleDetector(
                        model_path=_coco_path,
                        confidence=0.1,
                        device=self.config.get('device', 'cpu'),
                        imgsz=self.config.get('imgsz', 768),
                        enable_tiling=False,
                    )
                else:
                    # 本地没有 → 尝试通过 ultralytics 自动下载（首次会下载约 6 MB，后续缓存）
                    _coco_path = None
                    try:
                        from ultralytics import YOLO
                        _m = YOLO('yolo12n.pt')
                        _src = Path(_m.ckpt_path) if hasattr(_m, 'ckpt_path') and _m.ckpt_path else None
                        if _src and Path(_src).exists():
                            import shutil
                            _dst = Path('models/yolo12n.pt')
                            _dst.parent.mkdir(parents=True, exist_ok=True)
                            if not _dst.exists():
                                shutil.copy2(_src, _dst)
                            _coco_path = str(_dst)
                            _log.info("Auto-downloaded COCO model -> %s", _coco_path)
                    except Exception as _exc:
                        _log.warning("Cannot auto-download yolo12n.pt: %s", _exc)
                    if _coco_path:
                        _log.info("Primary model lacks traffic-light class; using %s for context detection", _coco_path)
                        self.context_detector = VehicleDetector(
                            model_path=_coco_path,
                            confidence=0.1,
                            device=self.config.get('device', 'cpu'),
                            imgsz=self.config.get('imgsz', 768),
                            enable_tiling=False,
                        )
                    else:
                        _log.warning(
                            "Traffic light detection unavailable — no COCO model found and auto-download failed. "
                            "Place yolo12n.pt in models/ to enable."
                        )

            self.tracker = ByteTracker(
                track_thresh=self.config.get('track_thresh', 0.5),
                track_buffer=self.config.get('track_buffer', 30),
                match_thresh=self.config.get('match_thresh', 0.8),
                min_box_area=self.config.get('min_box_area', 10),
            )

            self.feature_extractor = FeatureExtractor(
                pixel_to_meter=self.config.get('pixel_to_meter', 0.05),
                fps=self.config.get('fps', 15)
            )
            # 使用实际视频帧率覆盖配置文件值
            actual_fps = self.video_stream.get_fps()
            if actual_fps > 0:
                self.feature_extractor.set_fps(actual_fps)

            self.violation_detector = AdaptiveViolationDetector(
                speed_limit=self.config.get('speed_limit', 60),
                snapshot_dir=self.config.get('snapshot_dir', 'data/snapshots'),
                emergency_distance=self.config.get('emergency_distance', 300),
                red_light_enabled=self.config.get('red_light_enabled', True),
                speeding_enabled=self.config.get('speeding_enabled', True),
                wrong_way_enabled=self.config.get('wrong_way_enabled', True),
                illegal_lane_enabled=self.config.get('illegal_lane_enabled', True),
                expected_flow_direction=self.config.get('expected_flow_direction', 'south'),
                lane_change_lateral_px=self.config.get('lane_change_lateral_px', 80),
                lane_change_min_speed_kmh=self.config.get('lane_change_min_speed_kmh', 15),
                traffic_light_config=self.config.get('traffic_light_config'),
                emergency_vehicle_config=self.config.get('emergency_vehicle_config'),
                traffic_police_config=self.config.get('traffic_police_config'),
            )

            self.interaction_graph = None
            self.collision_predictor = None
            if self.config.get('enable_risk', True):
                self.interaction_graph = VehicleInteractionGraph(
                    distance_threshold=self.config.get('interaction_distance', 200),
                    temporal_window=self.config.get('risk_temporal_window', 10),
                    model_path=self.config.get('stgat_model_path'),
                    device=self.config.get('device', 'cpu'),
                )
                self.collision_predictor = CollisionRiskPredictor(
                    history_length=self.config.get('risk_history_length', 10),
                    prediction_horizon=self.config.get('risk_prediction_horizon', 15),
                    fps=self.config.get('fps', 15),
                    collision_threshold=self.config.get('risk_collision_threshold', 150.0),
                    ttc_thresholds=self.config.get('risk_ttc_thresholds'),
                    model_path=self.config.get('collision_model_path'),
                    device=self.config.get('device', 'cpu'),
                )

            # 车牌 OCR 延迟到首次需要时再加载，避免阻塞视频线程启动
            self.plate_reader = None
            self._ocr_enabled = bool(
                self.config.get('ocr_enabled', True)
                and int(self.config.get('ocr_interval', 0)) > 0
            )

            if self.config.get('stop_line'):
                sl = self.config['stop_line']
                self.violation_detector.set_stop_line(sl['y'], sl['x_start'], sl['x_end'])

            self.running = True
            frame_count = 0
            plate_cache = {}
            last_collision_risks = []
            ocr_scheduler = OCRScheduler(
                max_per_frame=int(self.config.get('ocr_max_vehicles_per_frame', 8))
            )
            base_risk_interval = int(self.config.get('risk_interval', 3))
            base_ocr_interval = int(self.config.get('ocr_interval', 10))
            ocr_min_h = int(self.config.get('ocr_min_bbox_height', 40))

            # GPU / 推理预热，避免冷启动 FPS 误触发降级
            if self.detector.device == "cuda":
                import numpy as _np
                _warm = _np.zeros((360, 640, 3), dtype=_np.uint8)
                self.detector.detect_vehicles(_warm)

            if perf_enabled and fps_monitor:
                fps_monitor.mark_ready()

            while self.running:
                # 暂停时休眠，保持线程存活
                while self._paused and self.running:
                    self.msleep(100)
                if not self.running:
                    break

                perf_enabled = self._runtime_perf_enabled
                perf = self._perf_optimizer
                fps_monitor = self._fps_monitor
                if self.video_stream is None or self.video_stream.cap is None:
                    break

                frame_count += 1
                if perf_enabled and perf is not None and not perf.should_process_frame(frame_count):
                    if not self.video_stream.grab():
                        break
                    continue

                frame_start = time.perf_counter()
                ret, frame = self.video_stream.read()
                if not ret or frame is None:
                    break

                if perf_enabled and perf is not None:
                    self.detector.imgsz = perf.get_imgsz()
                    self.detector.enable_tiling = perf.get_enable_tiling()
                elif self.detector is not None:
                    self.detector.imgsz = int(self.config.get('imgsz', 768))
                    self.detector.enable_tiling = bool(self.config.get('enable_tiling', False))

                effective_risk_interval = (
                    perf.get_risk_interval(base_risk_interval)
                    if perf_enabled and perf is not None else base_risk_interval
                )
                effective_ocr_interval = (
                    perf.get_ocr_interval(base_ocr_interval)
                    if perf_enabled and perf is not None else base_ocr_interval
                )
                risk_active = bool(self.config.get('enable_risk', True)) and not (
                    perf_enabled and perf is not None and perf.risk_disabled()
                )

                # ===== 1) 检测：单次推理为主，避免每帧多次 predict 导致卡顿 =====
                # 有停止线时必须检测交通灯状态，否则停止线始终显示为绿色
                enable_context = bool(
                    self.config.get('enable_context_detection', False)
                    or self.config.get('stop_line')
                )
                if enable_context:
                    # 车辆正常阈值，红绿灯和人用更低的置信度（红绿灯在画面中占比极小）
                    vehicle_dets = self.detector.detect_vehicles(frame)
                    ctx = self.context_detector if self.context_detector is not None else self.detector
                    ctx_dets = ctx.detect(
                        frame,
                        [ctx.PERSON_CLASS, ctx.TRAFFIC_LIGHT_CLASS],
                        conf=0.1,
                    )
                    person_bboxes = [d.bbox for d in ctx_dets if d.class_id == ctx.PERSON_CLASS]
                    light_dets = [d for d in ctx_dets if d.class_id == ctx.TRAFFIC_LIGHT_CLASS]
                    # 多灯路口：优先选离停止线最近的灯
                    from src.core.adaptive_violation import select_best_light_bbox
                    stop_y = self.violation_detector.stop_line.y if self.violation_detector.stop_line else None
                    fh, fw = frame.shape[:2]
                    light_bbox = select_best_light_bbox(light_dets, stop_line_y=stop_y, frame_h=fh, frame_w=fw)
                else:
                    vehicle_dets = self.detector.detect_vehicles(frame)
                    person_bboxes = []
                    light_bbox = None

                tracks = self.tracker.update(vehicle_dets)
                vehicle_bboxes = [t.bbox for t in tracks]

                self.violation_detector.update(
                    frame,
                    vehicle_bboxes,
                    person_bboxes=person_bboxes,
                    light_bbox=light_bbox,
                )

                # Prepare track data for ST-GAT and collision prediction
                track_data = [
                    {'track_id': t.track_id, 'bbox': t.bbox}
                    for t in tracks
                ]

                # ===== 2) 风险计算：降频 + 限制参与车辆数量，降低 O(n^2) 压力 =====
                risk_max_tracks = int(self.config.get('risk_max_tracks', 20))

                collision_risks = last_collision_risks
                if (
                    risk_active
                    and self.interaction_graph
                    and self.collision_predictor
                    and effective_risk_interval > 0
                ):
                    if frame_count % effective_risk_interval == 0:
                        # 取面积最大的前 N 辆车参与风险计算（远处小车对风险影响较小且计算成本高）
                        def _area(td):
                            x1, y1, x2, y2 = td['bbox']
                            return max(0, (x2 - x1) * (y2 - y1))

                        track_data_risk = sorted(track_data, key=_area, reverse=True)[:max(1, risk_max_tracks)]
                        self.interaction_graph.update(track_data_risk)
                        collision_risks = self.collision_predictor.update(track_data_risk)
                        last_collision_risks = collision_risks

                features_list = []
                violations = []
                plate_results = {}  # track_id -> plate_number

                ocr_targets = []
                if self._ocr_enabled and effective_ocr_interval > 0 and frame_count % effective_ocr_interval == 0:
                    bbox_heights = {t.track_id: t.bbox[3] - t.bbox[1] for t in tracks}
                    ocr_targets = ocr_scheduler.select_tracks(
                        [t.track_id for t in tracks],
                        plate_cache,
                        bbox_heights,
                        ocr_min_h,
                    )

                for track in tracks:
                    plate_number = plate_cache.get(track.track_id)
                    if self._ocr_enabled and track.track_id in ocr_targets:
                        if self.plate_reader is None:
                            self.plate_reader = PlateReader(
                                model_path=self.config.get('plate_model_path', 'models/plate_ocr.pt'),
                                use_gpu=self.config.get('device', 'cpu') != 'cpu',
                                paddle_mobile=self.config.get('ocr_paddle_mobile', True),
                            )
                        if self.plate_reader:
                            plate_result = self.plate_reader.read(frame, track.bbox)
                            if plate_result:
                                plate_number = plate_result.plate_number
                                plate_cache[track.track_id] = plate_number
                                plate_results[track.track_id] = plate_number
                            elif track.track_id not in plate_cache:
                                pending = self.config.get("plate_pending_label", "识别中")
                                plate_cache[track.track_id] = pending
                                plate_results[track.track_id] = pending
                    elif (
                        self.plate_reader
                        and plate_number
                        and plate_number != self.config.get("plate_pending_label", "识别中")
                    ):
                        plate_results[track.track_id] = plate_number

                    features = self.feature_extractor.extract(
                        frame, track.track_id, track.bbox
                    )
                    features_list.append(features)

                    record = self.violation_detector.check_violation(
                        track_id=track.track_id,
                        bbox=track.bbox,
                        speed=features.speed,
                        frame=frame,
                        plate_number=plate_number,
                        direction=features.direction,
                    )
                    if record:
                        violations.append(record)

                annotated_frame = self.violation_detector.draw_annotations(frame)
                if risk_active and collision_risks:
                    annotated_frame = self.collision_predictor.draw_predictions(
                        annotated_frame, collision_risks, track_data
                    )

                if perf_enabled and perf is not None and fps_monitor and frame_count % 30 == 0:
                    status = perf.get_status()
                    cv2.putText(
                        annotated_frame,
                        f"FPS:{fps_monitor.avg_fps:.1f} imgsz:{status['imgsz']} L{status['degradation_level']}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (200, 200, 200),
                        1,
                    )

                self.frame_ready.emit(annotated_frame, tracks, features_list, violations, collision_risks, plate_results)

                # 定期清理已消失 track 的历史（防止内存泄漏）
                if frame_count % 150 == 0:
                    active_ids = {t.track_id for t in tracks}
                    self.violation_detector.cleanup_stale_tracks(active_ids)
                    self.feature_extractor.cleanup_stale_tracks(active_ids)

                if frame_count % 30 == 0:
                    stats = self.violation_detector.get_statistics()
                    stats['emergency_vehicles'] = len(self.violation_detector.current_emergency_vehicles)
                    if risk_active and self.collision_predictor:
                        stats['collision_risks'] = self.collision_predictor.get_risk_summary(collision_risks)
                    else:
                        stats['collision_risks'] = {}
                    if perf_enabled and perf is not None and fps_monitor:
                        pstatus = perf.get_status()
                        stats['performance'] = {
                            'avg_fps': round(fps_monitor.avg_fps, 1),
                            'imgsz': pstatus['imgsz'],
                            'degradation_level': pstatus['degradation_level'],
                            'frame_skip': pstatus['frame_skip'],
                        }
                    self.stats_updated.emit(stats)

                if perf_enabled and perf is not None and fps_monitor:
                    fps_monitor.tick(time.perf_counter() - frame_start)
                    plan = fps_monitor.check_performance()
                    if plan:
                        perf.apply_degradation(plan)

        except Exception as e:
            import logging
            import traceback
            logging.getLogger(__name__).exception("Video thread error")
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
        finally:
            if self.video_stream is not None:
                self.video_stream.release()

    def pause(self):
        """Pause processing (keep thread alive)."""
        self._paused = True

    def resume(self):
        """Resume processing after pause."""
        self._paused = False

    def stop(self):
        """Stop processing"""
        self.running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)


class MainWindow(QMainWindow):
    """Main Window"""

    def __init__(self):
        super().__init__()
        self._tr = Translator("zh_CN")
        self._last_stats: Dict = {}
        self._last_collision_risks: list = []
        self.setGeometry(100, 100, 1560, 960)

        self._apply_global_style()

        self.video_thread: Optional[VideoThread] = None
        self.current_frame: Optional[np.ndarray] = None
        self._plate_cache: Dict[int, str] = {}
        self._seen_vehicle_tracks = set()
        # 交互优化：缓存最近违规记录，便于过滤切换即时刷新
        self._violation_cache: deque[ViolationRecord] = deque(maxlen=500)
        self._max_violation_rows = 200
        self._is_processing = False
        self._is_paused = False
        self._suppress_settings_apply = False
        # 防抖定时器：GUI 设置变更后 500ms 自动持久化到 settings.yaml
        self._settings_save_timer = QTimer(self)
        self._settings_save_timer.setSingleShot(True)
        self._settings_save_timer.setInterval(500)
        self._settings_save_timer.timeout.connect(self._persist_settings_to_yaml)

        self.config = {
            'fps': 15,
            'model_path': 'models/yolo12n_vehicle.pt',
            'plate_model_path': 'models/plate_ocr.pt',
            'stgat_model_path': None,
            'collision_model_path': None,
            'confidence': 0.2,
            'iou_threshold': 0.45,
            'device': 'cuda',
            'track_thresh': 0.5,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'min_box_area': 10,
            'pixel_to_meter': 0.05,
            'risk_ttc_thresholds': None,
            'speed_limit': 60,
            'snapshot_dir': 'data/snapshots',
            'emergency_distance': 300,
            'interaction_distance': 200,
            'risk_temporal_window': 10,
            'risk_history_length': 10,
            'risk_prediction_horizon': 15,
            'risk_collision_threshold': 150.0,
            # 性能相关开关
            'enable_context_detection': False,  # 人/信号灯上下文检测（会额外增加检测负担）
            'enable_risk': True,
            'risk_interval': 3,
            'risk_max_tracks': 20,
            'ocr_enabled': True,
            'ocr_paddle_mobile': True,
            'ocr_interval': 10,
            'ocr_min_bbox_height': 40,
            'ocr_max_vehicles_per_frame': 5,
            'performance_warmup_frames': 60,
            'performance_low_fps_checks': 5,
            # 小目标/远处车辆检测增强
            'imgsz': 768,
            'max_det': 300,
            'enable_tiling': False,
            'tiling_grid': (2, 2),
            'tiling_overlap': 0.20,
            'tiling_min_dets': 10,
            'stop_line': None,
            'wrong_way_enabled': True,
            'illegal_lane_enabled': True,
            'expected_flow_direction': 'south',
            'lane_change_lateral_px': 80,
            'lane_change_min_speed_kmh': 15,
            'tiling_interval': 5,
            'tiling_mode': 'strip',
            'performance_enabled': True,
            'performance_target_fps': 15,
            'performance_dynamic_resolution': True,
            'performance_frame_skip': 1,
            'database_path': 'data/traffic.db',
            'database_pool_size': 5,
            'gui_language': 'zh_CN',
        }

        self._merge_config_from_yaml()
        lang = self.config.get('gui_language', 'zh_CN')
        if lang not in SUPPORTED_LANGUAGES:
            lang = 'zh_CN'
        self._tr.set_language(lang)
        self.database = Database(
            db_path=self.config.get('database_path', 'data/traffic.db'),
            pool_size=int(self.config.get('database_pool_size', 5)),
        )
        self._db_cleanup = start_db_cleanup_from_config(
            self.database,
            load_config('config/settings.yaml') or {'database': {}},
        )
        self._init_ui()
        self._retranslate_ui()
        gui_cfg = (load_config('config/settings.yaml') or {}).get('gui', {})
        size = gui_cfg.get('window_size')
        if isinstance(size, (list, tuple)) and len(size) == 2:
            self.setGeometry(100, 100, int(size[0]), int(size[1]))
        self._apply_config_to_controls()
        self._sync_controls_for_state()

    def _merge_config_from_yaml(self, path: str = 'config/settings.yaml'):
        """从 settings.yaml 合并配置（存在则覆盖默认值）。"""
        cfg = load_config(path)
        if not cfg:
            return
        det = cfg.get('detector', {})
        vio = cfg.get('violation', {})
        ocr = cfg.get('ocr', {})
        risk = cfg.get('risk', {})
        video = cfg.get('video', {})
        perf = cfg.get('performance', {})
        db = cfg.get('database', {})
        tracker = cfg.get('tracker', {})
        feature = cfg.get('feature', {})
        gui = cfg.get('gui', {})
        tl = cfg.get('traffic_light', {})
        ev = cfg.get('emergency_vehicle', {})
        tp = cfg.get('traffic_police', {})
        self.config.update({
            'fps': video.get('fps', self.config['fps']),
            'model_path': det.get('model_path', self.config['model_path']),
            'confidence': det.get('confidence', self.config['confidence']),
            'iou_threshold': det.get('iou_threshold', self.config.get('iou_threshold', 0.45)),
            'device': cfg.get('system', {}).get('device', self.config['device']),
            'imgsz': det.get('imgsz', self.config['imgsz']),
            'max_det': det.get('max_det', self.config['max_det']),
            'enable_tiling': det.get('enable_tiling', self.config['enable_tiling']),
            'tiling_mode': det.get('tiling_mode', self.config.get('tiling_mode', 'strip')),
            'tiling_interval': det.get('tiling_interval', self.config.get('tiling_interval', 5)),
            'enable_context_detection': det.get('enable_context_detection', False),
            'vehicle_classes': det.get('classes'),
            'resize': video.get('resize'),
            'red_light_enabled': vio.get('red_light_enabled', True),
            'speeding_enabled': vio.get('speeding_enabled', True),
            'speed_limit': vio.get('speed_limit', self.config['speed_limit']),
            'emergency_distance': vio.get('emergency_distance', self.config['emergency_distance']),
            'snapshot_dir': vio.get('snapshot_dir', self.config['snapshot_dir']),
            'stop_line': vio.get('stop_line'),
            'wrong_way_enabled': vio.get('wrong_way_enabled', True),
            'illegal_lane_enabled': vio.get('illegal_lane_enabled', True),
            'expected_flow_direction': vio.get('expected_flow_direction', 'south'),
            'lane_change_lateral_px': vio.get('lane_change_lateral_px', 80),
            'lane_change_min_speed_kmh': vio.get('lane_change_min_speed_kmh', 15),
            'plate_model_path': ocr.get('model_path', self.config['plate_model_path']),
            'ocr_enabled': ocr.get('enabled', self.config.get('ocr_enabled', True)),
            'ocr_paddle_mobile': ocr.get('paddle_mobile', self.config.get('ocr_paddle_mobile', True)),
            'ocr_interval': ocr.get('interval', self.config['ocr_interval']),
            'ocr_min_bbox_height': ocr.get('min_bbox_height', self.config['ocr_min_bbox_height']),
            'ocr_max_vehicles_per_frame': ocr.get('max_vehicles_per_frame', self.config['ocr_max_vehicles_per_frame']),
            'performance_warmup_frames': perf.get('warmup_frames', self.config.get('performance_warmup_frames', 45)),
            'performance_low_fps_checks': perf.get('low_fps_checks', self.config.get('performance_low_fps_checks', 5)),
            'track_thresh': tracker.get('track_thresh', self.config['track_thresh']),
            'track_buffer': tracker.get('track_buffer', self.config['track_buffer']),
            'match_thresh': tracker.get('match_thresh', self.config.get('match_thresh', 0.8)),
            'min_box_area': tracker.get('min_box_area', self.config.get('min_box_area', 10)),
            'pixel_to_meter': feature.get('pixel_to_meter', self.config['pixel_to_meter']),
            'stgat_model_path': risk.get('stgat_model_path', self.config['stgat_model_path']),
            'collision_model_path': risk.get(
                'collision_model_path', self.config['collision_model_path'],
            ),
            'interaction_distance': risk.get(
                'interaction_distance', self.config['interaction_distance'],
            ),
            'risk_temporal_window': risk.get(
                'temporal_window', self.config['risk_temporal_window'],
            ),
            'risk_history_length': risk.get(
                'history_length', self.config['risk_history_length'],
            ),
            'risk_prediction_horizon': risk.get(
                'prediction_horizon', self.config['risk_prediction_horizon'],
            ),
            'risk_collision_threshold': risk.get(
                'collision_threshold', self.config['risk_collision_threshold'],
            ),
            'risk_ttc_thresholds': risk.get('ttc_thresholds', self.config.get('risk_ttc_thresholds')),
            'enable_risk': risk.get('enabled', self.config['enable_risk']),
            'risk_interval': risk.get('interval', self.config['risk_interval']),
            'risk_max_tracks': risk.get('max_tracks', self.config['risk_max_tracks']),
            'tiling_grid': tuple(det.get('tiling_grid', self.config.get('tiling_grid', (2, 2)))),
            'tiling_overlap': det.get('tiling_overlap', self.config.get('tiling_overlap', 0.20)),
            'tiling_min_dets': det.get('tiling_min_dets', self.config.get('tiling_min_dets', 10)),
            'performance_enabled': perf.get('enabled', self.config.get('performance_enabled', True)),
            'performance_target_fps': perf.get('target_fps', self.config.get('performance_target_fps', 15)),
            'performance_dynamic_resolution': perf.get(
                'dynamic_resolution', self.config.get('performance_dynamic_resolution', True),
            ),
            'performance_frame_skip': perf.get('frame_skip', self.config.get('performance_frame_skip', 1)),
            'database_path': db.get('path', self.config.get('database_path', 'data/traffic.db')),
            'database_pool_size': db.get('pool_size', self.config.get('database_pool_size', 5)),
            'gui_language': gui.get('language', self.config.get('gui_language', 'zh_CN')),
            'traffic_light_config': tl,
            'emergency_vehicle_config': ev,
            'traffic_police_config': tp,
        })

    def _apply_config_to_controls(self):
        """将 config 同步到设置面板控件。"""
        self._suppress_settings_apply = True
        try:
            self.spin_confidence.setValue(int(self.config.get('confidence', 0.2) * 100))
            self.spin_speed_limit.setValue(int(self.config.get('speed_limit', 60)))
            self.spin_emergency_dist.setValue(int(self.config.get('emergency_distance', 300)))
            idx = self.combo_flow_direction.findText(self.config.get('expected_flow_direction', 'south'))
            if idx >= 0:
                self.combo_flow_direction.setCurrentIndex(idx)
            self.cb_enable_tiling.setChecked(bool(self.config.get('enable_tiling', False)))
            self.cb_wrong_way.setChecked(bool(self.config.get('wrong_way_enabled', True)))
            self.cb_illegal_lane.setChecked(bool(self.config.get('illegal_lane_enabled', True)))
            self.cb_performance.setChecked(bool(self.config.get('performance_enabled', True)))
            sl = self.config.get('stop_line')
            if sl:
                self.cb_enable_stopline.setChecked(True)
                self.spin_sl_y.setValue(int(sl.get('y', 430)))
                self.spin_sl_x1.setValue(int(sl.get('x_start', 200)))
                self.spin_sl_x2.setValue(int(sl.get('x_end', 1100)))
            else:
                self.cb_enable_stopline.setChecked(False)
        finally:
            self._suppress_settings_apply = False

    def _collect_detection_settings_from_ui(self) -> dict:
        """Read detection-related settings from the settings panel."""
        settings = {
            'confidence': self.spin_confidence.value() / 100,
            'speed_limit': self.spin_speed_limit.value(),
            'emergency_distance': self.spin_emergency_dist.value(),
            'expected_flow_direction': self.combo_flow_direction.currentText(),
            'enable_tiling': self.cb_enable_tiling.isChecked(),
            'wrong_way_enabled': self.cb_wrong_way.isChecked(),
            'illegal_lane_enabled': self.cb_illegal_lane.isChecked(),
            'performance_enabled': self.cb_performance.isChecked(),
            'performance_target_fps': self.config.get('fps', 15),
        }
        if self.cb_enable_stopline.isChecked():
            settings['stop_line'] = {
                'y': self.spin_sl_y.value(),
                'x_start': self.spin_sl_x1.value(),
                'x_end': self.spin_sl_x2.value(),
            }
        else:
            settings['stop_line'] = None
        return settings

    def _connect_runtime_setting_controls(self) -> None:
        """Toggle detection features immediately when settings change."""
        for checkbox in (
            self.cb_enable_tiling,
            self.cb_wrong_way,
            self.cb_illegal_lane,
            self.cb_performance,
            self.cb_enable_stopline,
        ):
            checkbox.stateChanged.connect(self._on_detection_settings_changed)

        for spinbox in (
            self.spin_confidence,
            self.spin_speed_limit,
            self.spin_emergency_dist,
            self.spin_sl_y,
            self.spin_sl_x1,
            self.spin_sl_x2,
        ):
            spinbox.valueChanged.connect(self._on_detection_settings_changed)

        self.combo_flow_direction.currentIndexChanged.connect(self._on_detection_settings_changed)

    def _on_detection_settings_changed(self, *_args) -> None:
        if self._suppress_settings_apply:
            return
        self._push_detection_settings_to_runtime()

    def _push_detection_settings_to_runtime(self) -> None:
        """Sync settings panel values into config and the active video thread."""
        settings = self._collect_detection_settings_from_ui()
        self.config.update(settings)

        running = bool(self.video_thread and self.video_thread.isRunning())
        if running:
            self.video_thread.apply_runtime_settings(settings)
            self.statusBar.showMessage(self._tr.tr("status_settings_applied"), 3000)
        elif self._is_processing:
            self.statusBar.showMessage(self._tr.tr("status_settings_saved"), 2000)

        # 防抖持久化：500ms 内无新变更则自动写入 settings.yaml
        self._settings_save_timer.start()

    def _persist_settings_to_yaml(self) -> None:
        """将当前设置面板的值批量写入 settings.yaml（防抖回调）。"""
        s = self._collect_detection_settings_from_ui()
        config_path = 'config/settings.yaml'

        # 映射: (section, key) -> 从 s 中取值的键
        key_map = [
            ('detector',   'confidence',            s['confidence']),
            ('detector',   'enable_tiling',         s['enable_tiling']),
            ('violation',  'speed_limit',           s['speed_limit']),
            ('violation',  'emergency_distance',     s['emergency_distance']),
            ('violation',  'expected_flow_direction', s['expected_flow_direction']),
            ('violation',  'wrong_way_enabled',     s['wrong_way_enabled']),
            ('violation',  'illegal_lane_enabled',  s['illegal_lane_enabled']),
            ('violation',  'stop_line',             s['stop_line']),
            ('performance','enabled',               s['performance_enabled']),
            ('performance','target_fps',            s['performance_target_fps']),
        ]

        for section, key, value in key_map:
            if value is None:
                continue
            save_config_section(config_path, section, key, value)

    def _apply_global_style(self):
        """应用全局深色主题（见 src/gui/theme.py）。"""
        palette = self.palette()
        palette.setColor(palette.Window, QColor(COLORS["bg"]))
        palette.setColor(palette.Base, QColor(COLORS["bg_elevated"]))
        palette.setColor(palette.AlternateBase, QColor(COLORS["surface"]))
        palette.setColor(palette.Text, QColor(COLORS["text"]))
        palette.setColor(palette.WindowText, QColor(COLORS["text"]))
        palette.setColor(palette.Button, QColor(COLORS["surface"]))
        palette.setColor(palette.ButtonText, QColor(COLORS["text"]))
        palette.setColor(palette.Highlight, QColor(COLORS["accent"]))
        palette.setColor(palette.HighlightedText, QColor(COLORS["bg"]))
        self.setPalette(palette)
        self.setStyleSheet(global_stylesheet())

    def _set_live_badge(self, running: bool) -> None:
        if not getattr(self, "live_badge", None):
            return
        if running:
            self.live_badge.setText(self._tr.tr("live_running"))
            self.live_badge.setStyleSheet(
                f"background-color: {COLORS['text']};"
                f"color: {COLORS['bg']};"
                f"border: 1px solid {COLORS['text']};"
                f"border-radius: 4px; padding: 4px 10px;"
                f"font-size: 11px; font-weight: 700;"
            )
        else:
            self.live_badge.setText(self._tr.tr("live_idle"))
            self.live_badge.setObjectName("LiveBadge")
            self.live_badge.setStyleSheet("")

    def _on_language_changed(self, index: int) -> None:
        lang = "zh_CN" if index == 0 else "en"
        self._tr.set_language(lang)
        self.config["gui_language"] = lang
        self._retranslate_ui()
        self._sync_controls_for_state()
        self._set_live_badge(self._is_processing)
        if self._last_stats:
            self._on_stats_updated(self._last_stats)

    def _retranslate_ui(self) -> None:
        tr = self._tr
        self.setWindowTitle(tr.tr("app_title"))
        self.header_title.setText(tr.tr("app_title"))
        self.header_subtitle.setText(tr.tr("app_subtitle"))
        self.lang_label.setText(tr.tr("language"))
        self.combo_language.blockSignals(True)
        self.combo_language.setItemText(0, tr.tr("lang_zh"))
        self.combo_language.setItemText(1, tr.tr("lang_en"))
        self.combo_language.blockSignals(False)
        self.video_label.setText(tr.tr("video_placeholder"))
        self.btn_open.setText(tr.tr("btn_open_video"))
        self.btn_camera.setText(tr.tr("btn_open_camera"))
        self.btn_stop.setText(tr.tr("btn_stop"))
        self.stats_group.setTitle(tr.tr("group_realtime"))
        self.card_vehicles.set_title(tr.tr("card_vehicles"))
        self.card_speed.set_title(tr.tr("card_speed"))
        self.card_emergency.set_title(tr.tr("card_emergency"))
        self.card_fps.set_title(tr.tr("card_fps"))
        self.vf_title.setText(tr.tr("violations_title"))
        self.rf_title.setText(tr.tr("risk_title"))
        self.vehicle_group.setTitle(tr.tr("group_vehicles"))
        self.vehicle_table.setHorizontalHeaderLabels([
            tr.tr("veh_header_id"), tr.tr("veh_header_type"), tr.tr("veh_header_color"),
            tr.tr("veh_header_speed"), tr.tr("veh_header_dir"), tr.tr("veh_header_plate"),
        ])
        self.cb_show_exempted.setText(tr.tr("show_exempted"))
        self.cb_only_exempted.setText(tr.tr("only_exempted"))
        self.violation_table.setHorizontalHeaderLabels([
            tr.tr("vio_header_time"), tr.tr("vio_header_type"), tr.tr("vio_header_plate"),
            tr.tr("vio_header_speed"), tr.tr("vio_header_status"),
            tr.tr("vio_header_reason"), tr.tr("vio_header_details"),
        ])
        self.search_input.setPlaceholderText(tr.tr("search_plate_ph"))
        self.btn_search.setText(tr.tr("btn_search"))
        self.db_table_label.setText(tr.tr("db_table_label"))
        self.db_search_input.setPlaceholderText(tr.tr("db_search_ph"))
        self.db_refresh_btn.setText(tr.tr("db_refresh"))
        self.db_delete_btn.setText(tr.tr("db_delete"))
        self.db_clean_label.setText(tr.tr("db_clean_label"))
        self.db_clean_btn.setText(tr.tr("db_clean_btn"))
        self.exemption_info.setText(tr.tr("exemption_html"))
        self.btn_reset_stats.setText(tr.tr("btn_reset_stats"))
        self.detect_group.setTitle(tr.tr("group_detection"))
        self.lbl_confidence.setText(tr.tr("label_confidence"))
        self.lbl_speed_limit.setText(tr.tr("label_speed_limit"))
        self.lbl_emergency_dist.setText(tr.tr("label_emergency_dist"))
        self.lbl_flow_dir.setText(tr.tr("label_flow_dir"))
        self.cb_enable_tiling.setText(tr.tr("cb_tiling"))
        self.cb_wrong_way.setText(tr.tr("cb_wrong_way"))
        self.cb_illegal_lane.setText(tr.tr("cb_illegal_lane"))
        self.cb_performance.setText(tr.tr("cb_performance"))
        self.btn_reload_cfg.setText(tr.tr("btn_reload_cfg"))
        self.stopline_group.setTitle(tr.tr("group_stopline"))
        self.cb_enable_stopline.setText(tr.tr("cb_stopline"))
        self.lbl_sl_y.setText(tr.tr("label_sl_y"))
        self.lbl_sl_x.setText(tr.tr("label_sl_x"))
        self.btn_auto_stopline.setText(tr.tr("btn_auto_stopline"))
        self.right_panel.setTabText(0, tr.tr("tab_realtime"))
        self.right_panel.setTabText(1, tr.tr("tab_violations"))
        self.right_panel.setTabText(2, tr.tr("tab_database"))
        self.right_panel.setTabText(3, tr.tr("tab_exemption"))
        self.right_panel.setTabText(4, tr.tr("tab_statistics"))
        self.right_panel.setTabText(5, tr.tr("tab_settings"))
        if self.stats_canvas:
            self.stats_canvas.set_translator(tr)
        if not self._is_processing:
            self.statusBar.showMessage(tr.tr("status_ready"))
        self._refresh_violation_labels()
        self._refresh_risk_labels()

    def _init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(16, 16, 16, 12)
        root_layout.setSpacing(14)

        app_header, self.header_title, self.header_subtitle, self.live_badge, header_right = build_app_header(self._tr)
        self.lang_label = QLabel()
        self.lang_label.setObjectName("LangLabel")
        self.combo_language = QComboBox()
        self.combo_language.addItem(self._tr.tr("lang_zh"), "zh_CN")
        self.combo_language.addItem(self._tr.tr("lang_en"), "en")
        lang_idx = 0 if self._tr.language == "zh_CN" else 1
        self.combo_language.setCurrentIndex(lang_idx)
        self.combo_language.currentIndexChanged.connect(self._on_language_changed)
        header_right.insertWidget(0, self.combo_language)
        header_right.insertWidget(0, self.lang_label)
        root_layout.addWidget(app_header)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(14)

        # Left panel: Video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        video_frame = QFrame()
        video_frame.setObjectName("VideoPanel")
        video_frame_layout = QVBoxLayout(video_frame)
        video_frame_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setObjectName("VideoPlaceholder")
        self.video_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding,
        )
        self.video_label.setScaledContents(False)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_frame_layout.addWidget(self.video_label)
        left_layout.addWidget(video_frame, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.btn_open = QPushButton()
        self.btn_camera = QPushButton()
        self.btn_stop = QPushButton()
        self.btn_stop.setEnabled(False)
        for btn in (self.btn_open, self.btn_camera):
            btn.setProperty("btnRole", "primary")
            btn.setMinimumHeight(42)
        self.btn_stop.setProperty("btnRole", "danger")
        self.btn_stop.setMinimumHeight(42)
        for btn in (self.btn_open, self.btn_camera, self.btn_stop):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        self.btn_open.clicked.connect(self._open_video)
        self.btn_camera.clicked.connect(self._open_camera)
        self.btn_stop.clicked.connect(self._toggle_pause)

        btn_layout.addWidget(self.btn_open, stretch=1)
        btn_layout.addWidget(self.btn_camera, stretch=1)
        btn_layout.addWidget(self.btn_stop, stretch=1)
        left_layout.addLayout(btn_layout)

        self.right_panel = QTabWidget()
        right_panel = self.right_panel
        right_panel.setMinimumWidth(420)
        right_panel.setMaximumWidth(560)
        right_panel.setDocumentMode(True)
        right_panel.setUsesScrollButtons(True)

        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)
        info_layout.setSpacing(12)

        self.stats_group = QGroupBox()
        stats_group = self.stats_group
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(10)

        metrics_row1 = QHBoxLayout()
        self.card_vehicles = MetricCard("", "0")
        self.card_speed = MetricCard("", "0 km/h")
        metrics_row1.addWidget(self.card_vehicles)
        metrics_row1.addWidget(self.card_speed)

        metrics_row2 = QHBoxLayout()
        self.card_emergency = MetricCard("", "0")
        self.card_fps = MetricCard("", "--")
        metrics_row2.addWidget(self.card_emergency)
        metrics_row2.addWidget(self.card_fps)
        stats_layout.addLayout(metrics_row1)
        stats_layout.addLayout(metrics_row2)

        self.label_perf_degrade_value = QLabel("推理 · imgsz -- · 降级 L0 · 跳帧 ×1")
        self.label_perf_degrade_value.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px; padding: 0 4px;")
        stats_layout.addWidget(self.label_perf_degrade_value)

        violation_frame = QFrame()
        violation_frame.setStyleSheet(insight_panel_style())
        vf_layout = QVBoxLayout(violation_frame)
        self.vf_title = QLabel()
        vf_title = self.vf_title
        vf_title.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 600; font-size: 11px;")
        self.label_total_violations = QLabel("累计违规：0")
        self.label_actual_violations = QLabel("实际违规：0")
        self.label_actual_violations.setStyleSheet(f"color: {COLORS['text']}; font-weight: 700;")
        self.label_exempted = QLabel()
        self.label_exempted.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 700;")
        vf_layout.addWidget(vf_title)
        vf_layout.addWidget(self.label_total_violations)
        vf_layout.addWidget(self.label_actual_violations)
        vf_layout.addWidget(self.label_exempted)
        stats_layout.addWidget(violation_frame)

        risk_frame = QFrame()
        risk_frame.setStyleSheet(insight_panel_style())
        rf_layout = QVBoxLayout(risk_frame)
        self.rf_title = QLabel()
        self.rf_title.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 600; font-size: 11px;")
        rf_title = self.rf_title
        self.label_collision_risk = QLabel()
        self.label_collision_risk.setStyleSheet(f"color: {COLORS['text']}; font-weight: 700; font-size: 14px;")
        self.label_min_ttc = QLabel("最短 TTC：--")
        self.label_risk_count = QLabel("活跃风险对：0")
        rf_layout.addWidget(rf_title)
        rf_layout.addWidget(self.label_collision_risk)
        rf_layout.addWidget(self.label_min_ttc)
        rf_layout.addWidget(self.label_risk_count)
        stats_layout.addWidget(risk_frame)

        info_layout.addWidget(stats_group)

        self.vehicle_group = QGroupBox()
        vehicle_group = self.vehicle_group
        vehicle_layout = QVBoxLayout(vehicle_group)
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(6)
        self.vehicle_table.horizontalHeader().setStretchLastSection(True)
        self.vehicle_table.verticalHeader().setVisible(False)
        self.vehicle_table.setShowGrid(False)
        self.vehicle_table.setAlternatingRowColors(True)
        self.vehicle_table.setSelectionBehavior(self.vehicle_table.SelectRows)
        self.vehicle_table.setEditTriggers(self.vehicle_table.NoEditTriggers)
        vehicle_layout.addWidget(self.vehicle_table)
        info_layout.addWidget(vehicle_group)

        right_panel.addTab(info_tab, "")

        violation_tab = QWidget()
        violation_layout = QVBoxLayout(violation_tab)

        filter_layout = QHBoxLayout()
        self.cb_show_exempted = QCheckBox()
        self.cb_show_exempted.setChecked(True)
        self.cb_only_exempted = QCheckBox()
        filter_layout.addWidget(self.cb_show_exempted)
        filter_layout.addWidget(self.cb_only_exempted)
        filter_layout.addStretch()
        violation_layout.addLayout(filter_layout)

        self.violation_table = QTableWidget()
        self.violation_table.setColumnCount(7)
        self.violation_table.horizontalHeader().setStretchLastSection(True)
        self.violation_table.setAlternatingRowColors(True)
        self.violation_table.verticalHeader().setVisible(False)
        self.violation_table.setShowGrid(False)
        self.violation_table.setSelectionBehavior(self.violation_table.SelectRows)
        self.violation_table.setEditTriggers(self.violation_table.NoEditTriggers)
        violation_layout.addWidget(self.violation_table)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.btn_search = QPushButton()
        self.btn_search.setProperty("btnRole", "primary")
        self.btn_search.style().unpolish(self.btn_search)
        self.btn_search.style().polish(self.btn_search)
        self.btn_search.clicked.connect(self._search_plate)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.btn_search)
        violation_layout.addLayout(search_layout)
        # 交互优化：切换过滤立即刷新
        self.cb_show_exempted.stateChanged.connect(self._refresh_violation_table)
        self.cb_only_exempted.stateChanged.connect(self._refresh_violation_table)

        right_panel.addTab(violation_tab, "")

        # ===== Vehicles DB Tab =====
        vehicles_tab = QWidget()
        vehicles_layout = QVBoxLayout(vehicles_tab)

        toolbar_layout = QHBoxLayout()
        self.db_table_combo = QComboBox()
        self.db_table_combo.addItems(["traffic_flow", "vehicles", "violations"])
        self.db_search_input = QLineEdit()
        self.db_search_input = QLineEdit()
        self.db_refresh_btn = QPushButton()
        self.db_delete_btn = QPushButton()
        self.db_table_label = QLabel()
        toolbar_layout.addWidget(self.db_table_label)
        toolbar_layout.addWidget(self.db_table_combo)
        toolbar_layout.addWidget(self.db_search_input)
        toolbar_layout.addWidget(self.db_refresh_btn)
        toolbar_layout.addWidget(self.db_delete_btn)
        vehicles_layout.addLayout(toolbar_layout)

        self.db_table = QTableWidget()
        self.db_table.horizontalHeader().setStretchLastSection(True)
        self.db_table.verticalHeader().setVisible(False)
        self.db_table.setShowGrid(False)
        self.db_table.setAlternatingRowColors(True)
        self.db_table.setSelectionBehavior(self.db_table.SelectRows)
        self.db_table.setEditTriggers(self.db_table.NoEditTriggers)
        vehicles_layout.addWidget(self.db_table)

        # 底部：清空当前所选表的所有记录
        clean_layout = QHBoxLayout()
        self.db_clean_label = QLabel()
        clean_layout.addWidget(self.db_clean_label)
        self.db_clean_btn = QPushButton()
        clean_layout.addWidget(self.db_clean_btn)
        clean_layout.addStretch()
        vehicles_layout.addLayout(clean_layout)

        # 绑定事件
        self.db_refresh_btn.clicked.connect(self._refresh_db_table)
        self.db_delete_btn.clicked.connect(self._delete_selected_db_vehicles)
        self.db_clean_btn.clicked.connect(self._clean_old_db_records)
        self.db_table_combo.currentIndexChanged.connect(self._refresh_db_table)

        right_panel.addTab(vehicles_tab, "")

        # ===== Special Cases Info Tab =====
        exemption_tab = QWidget()
        exemption_layout = QVBoxLayout(exemption_tab)

        self.exemption_info = QLabel()
        self.exemption_info.setWordWrap(True)
        self.exemption_info.setStyleSheet(
            f"padding: 16px; background-color: {COLORS['surface']};"
            f"border-radius: 8px; border: 1px solid {COLORS['border']};"
            f"color: {COLORS['text']};"
        )
        exemption_layout.addWidget(self.exemption_info)
        exemption_layout.addStretch()

        right_panel.addTab(exemption_tab, "")

        stats_tab = QWidget()
        stats_tab_layout = QVBoxLayout(stats_tab)
        self.stats_canvas = StatisticsCanvas(stats_tab, translator=self._tr)
        stats_tab_layout.addWidget(self.stats_canvas)

        self.btn_reset_stats = QPushButton()
        self.btn_reset_stats.clicked.connect(self._reset_statistics)
        stats_tab_layout.addWidget(self.btn_reset_stats)

        right_panel.addTab(stats_tab, "")

        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        self.detect_group = QGroupBox()
        detect_group = self.detect_group
        detect_layout = QVBoxLayout(detect_group)

        conf_layout = QHBoxLayout()
        self.lbl_confidence = QLabel()
        conf_layout.addWidget(self.lbl_confidence)
        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(1, 100)
        self.spin_confidence.setValue(20)
        self.spin_confidence.setSuffix("%")
        conf_layout.addWidget(self.spin_confidence)
        detect_layout.addLayout(conf_layout)

        speed_layout = QHBoxLayout()
        self.lbl_speed_limit = QLabel()
        speed_layout.addWidget(self.lbl_speed_limit)
        self.spin_speed_limit = QSpinBox()
        self.spin_speed_limit.setRange(1, 200)
        self.spin_speed_limit.setValue(60)
        self.spin_speed_limit.setSuffix(" km/h")
        speed_layout.addWidget(self.spin_speed_limit)
        detect_layout.addLayout(speed_layout)

        emergency_layout = QHBoxLayout()
        self.lbl_emergency_dist = QLabel()
        emergency_layout.addWidget(self.lbl_emergency_dist)
        self.spin_emergency_dist = QSpinBox()
        self.spin_emergency_dist.setRange(50, 1000)
        self.spin_emergency_dist.setValue(300)
        self.spin_emergency_dist.setSuffix(" px")
        emergency_layout.addWidget(self.spin_emergency_dist)
        detect_layout.addLayout(emergency_layout)

        flow_layout = QHBoxLayout()
        self.lbl_flow_dir = QLabel()
        flow_layout.addWidget(self.lbl_flow_dir)
        self.combo_flow_direction = QComboBox()
        self.combo_flow_direction.addItems([
            'north', 'south', 'east', 'west',
            'northeast', 'northwest', 'southeast', 'southwest',
        ])
        self.combo_flow_direction.setCurrentText('south')
        flow_layout.addWidget(self.combo_flow_direction)
        detect_layout.addLayout(flow_layout)

        self.cb_enable_tiling = QCheckBox()
        self.cb_enable_tiling.setChecked(False)
        detect_layout.addWidget(self.cb_enable_tiling)

        self.cb_wrong_way = QCheckBox()
        self.cb_wrong_way.setChecked(True)
        detect_layout.addWidget(self.cb_wrong_way)

        self.cb_illegal_lane = QCheckBox()
        self.cb_illegal_lane.setChecked(True)
        detect_layout.addWidget(self.cb_illegal_lane)

        self.cb_performance = QCheckBox()
        self.cb_performance.setChecked(True)
        detect_layout.addWidget(self.cb_performance)

        self.btn_reload_cfg = QPushButton()
        self.btn_reload_cfg.clicked.connect(self._reload_config_and_ui)
        detect_layout.addWidget(self.btn_reload_cfg)

        settings_layout.addWidget(detect_group)

        self.stopline_group = QGroupBox()
        stopline_group = self.stopline_group
        stopline_layout = QVBoxLayout(stopline_group)

        self.cb_enable_stopline = QCheckBox()
        stopline_layout.addWidget(self.cb_enable_stopline)

        sl_y_layout = QHBoxLayout()
        self.lbl_sl_y = QLabel()
        sl_y_layout.addWidget(self.lbl_sl_y)
        self.spin_sl_y = QSpinBox()
        self.spin_sl_y.setRange(0, 2000)
        self.spin_sl_y.setValue(400)
        sl_y_layout.addWidget(self.spin_sl_y)
        stopline_layout.addLayout(sl_y_layout)

        sl_x_layout = QHBoxLayout()
        self.lbl_sl_x = QLabel()
        sl_x_layout.addWidget(self.lbl_sl_x)
        self.spin_sl_x1 = QSpinBox()
        self.spin_sl_x1.setRange(0, 2000)
        self.spin_sl_x1.setValue(100)
        self.spin_sl_x2 = QSpinBox()
        self.spin_sl_x2.setRange(0, 2000)
        self.spin_sl_x2.setValue(500)
        sl_x_layout.addWidget(self.spin_sl_x1)
        sl_x_layout.addWidget(QLabel("-"))
        sl_x_layout.addWidget(self.spin_sl_x2)
        stopline_layout.addLayout(sl_x_layout)

        # 自动识别停止线按钮
        self.btn_auto_stopline = QPushButton()
        self.btn_auto_stopline.clicked.connect(self._auto_detect_stop_line)
        stopline_layout.addWidget(self.btn_auto_stopline)

        settings_layout.addWidget(stopline_group)
        settings_layout.addStretch()

        self._connect_runtime_setting_controls()

        right_panel.addTab(settings_tab, "")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        body_layout.addWidget(splitter)
        root_layout.addLayout(body_layout, stretch=1)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(self._tr.tr("status_ready"))

    def _reload_config_and_ui(self) -> None:
        self._merge_config_from_yaml()
        lang = self.config.get("gui_language", "zh_CN")
        if lang in SUPPORTED_LANGUAGES:
            self._tr.set_language(lang)
            self.combo_language.blockSignals(True)
            self.combo_language.setCurrentIndex(0 if lang == "zh_CN" else 1)
            self.combo_language.blockSignals(False)
        self._apply_config_to_controls()
        self._retranslate_ui()
        # 若视频正在运行，将重载的配置立即推送到检测管线
        self._push_detection_settings_to_runtime()

    def _refresh_violation_labels(self) -> None:
        tr = self._tr
        stats = self._last_stats
        self.label_total_violations.setText(tr.tr("total_violations", n=stats.get("total_violations", 0)))
        self.label_actual_violations.setText(tr.tr("actual_violations", n=stats.get("actual_violations", 0)))
        self.label_exempted.setText(tr.tr("exempted", n=stats.get("exempted_count", 0)))

    def _refresh_risk_labels(self) -> None:
        tr = self._tr
        risks = self._last_collision_risks
        if risks:
            highest = risks[0]
            level = tr.risk_level_label(highest.risk_level.value)
            self.label_collision_risk.setText(tr.tr("risk_level", level=level))
            if highest.time_to_collision > 0:
                unit = tr.tr("sec_unit")
                self.label_min_ttc.setText(
                    tr.tr("min_ttc", s=f"{highest.time_to_collision:.1f} {unit}")
                )
            else:
                self.label_min_ttc.setText(tr.tr("min_ttc_none"))
            self.label_risk_count.setText(tr.tr("active_risks", n=len(risks)))
        else:
            self.label_collision_risk.setText(tr.tr("risk_level", level=tr.tr("risk_safe")))
            self.label_min_ttc.setText(tr.tr("min_ttc_none"))
            self.label_risk_count.setText(tr.tr("active_risks", n=0))

    def _open_video(self):
        """Open video file"""
        tr = self._tr
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr.tr("dialog_select_video"), "",
            tr.tr("filter_video"),
        )
        if file_path:
            self._start_processing(file_path)

    def _open_camera(self):
        """Open camera"""
        self._start_processing("0")

    def _start_processing(self, source: str):
        """Start processing"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()

        self._is_processing = True
        self._sync_controls_for_state()

        self.config.update(self._collect_detection_settings_from_ui())
        self.config['plate_pending_label'] = self._tr.plate_pending_token()
        self.config['gui_language'] = self._tr.language

        self._plate_cache = {}
        self._seen_vehicle_tracks = set()

        # 始终尝试自动检测停止线，成功则覆盖 YAML 默认值
        self._try_auto_detect_stop_line(source)

        self.video_thread = VideoThread(source, self.config)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.stats_updated.connect(self._on_stats_updated)
        self.video_thread.error.connect(self._on_error)
        self.video_thread.start()

        self._set_live_badge(True)
        self.statusBar.showMessage(self._tr.tr("status_processing"))

    def _auto_detect_stop_line(self):
        """基于当前画面自动估计停止线位置，并填入设置面板。"""
        def _styled_info(title: str, text: str):
            box = QMessageBox(self)
            box.setWindowTitle(title)
            box.setText(text)
            box.setIcon(QMessageBox.Information)
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.setStyleSheet(message_box_stylesheet())
            box.exec_()

        if self.current_frame is None:
            _styled_info(self._tr.tr("stopline_title"), self._tr.tr("stopline_no_frame"))
            return

        frame = self.current_frame.copy()
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            _styled_info(self._tr.tr("stopline_title"), self._tr.tr("stopline_bad_frame"))
            return

        result = detect_stop_line(frame)
        if result is None:
            _styled_info(self._tr.tr("stopline_title"), self._tr.tr("stopline_not_found"))
            return

        y_global, x_start, x_end = result

        self._suppress_settings_apply = True
        try:
            self.spin_sl_y.setValue(y_global)
            self.spin_sl_x1.setValue(x_start)
            self.spin_sl_x2.setValue(x_end)
            self.cb_enable_stopline.setChecked(True)
        finally:
            self._suppress_settings_apply = False

        self._push_detection_settings_to_runtime()

        # 持久化：写入 settings.yaml，避免重载时回退
        save_config_section(
            'config/settings.yaml', 'violation', 'stop_line',
            {'y': y_global, 'x_start': x_start, 'x_end': x_end},
        )

        # 友好提示
        _styled_info(
            self._tr.tr("stopline_title"),
            self._tr.tr("stopline_ok", y=y_global, x1=x_start, x2=x_end),
        )

    def _try_auto_detect_stop_line(self, source: str):
        """启动处理前自动检测停止线（静默模式，多帧尝试，不弹窗）。"""
        import logging
        _log = logging.getLogger(__name__)
        import cv2 as _cv2
        try:
            cap = _cv2.VideoCapture(int(source) if str(source).isdigit() else source)
            if not cap.isOpened():
                _log.debug("Auto stop-line: cannot open source %s", source)
                return
            # 尝试多帧：摄像头首帧常为黑屏，视频首帧可能没有停止线
            result = None
            for _ in range(30):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                result = detect_stop_line(frame)
                if result is not None:
                    break
            cap.release()
            if result is None:
                _log.debug("Auto stop-line: not found in first 30 frames of %s", source)
                return
            y, x1, x2 = result
            self.config['stop_line'] = {'y': y, 'x_start': x1, 'x_end': x2}
            # 持久化：写入 settings.yaml，避免重载时回退
            save_config_section(
                'config/settings.yaml', 'violation', 'stop_line',
                {'y': y, 'x_start': x1, 'x_end': x2},
            )
            self._suppress_settings_apply = True
            try:
                self.spin_sl_y.setValue(y)
                self.spin_sl_x1.setValue(x1)
                self.spin_sl_x2.setValue(x2)
                self.cb_enable_stopline.setChecked(True)
            finally:
                self._suppress_settings_apply = False
            _log.info("Auto stop-line detected: y=%d x=[%d,%d]", y, x1, x2)
        except Exception:
            _log.debug("Auto stop-line detection failed", exc_info=True)

    def _toggle_pause(self):
        """暂停 / 继续 切换"""
        if not self.video_thread or not self.video_thread.isRunning():
            return
        if self._is_paused:
            self.video_thread.resume()
            self._is_paused = False
            self._set_live_badge(True)
            self.statusBar.showMessage(self._tr.tr("status_processing"))
        else:
            self.video_thread.pause()
            self._is_paused = True
            self._set_live_badge(False)
            self.statusBar.showMessage(self._tr.tr("status_paused"))
        self._sync_controls_for_state()

    def _stop_video(self):
        """Stop processing (complete stop, release resources)."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self._is_processing = False
        self._is_paused = False
        self._plate_cache = {}
        self._seen_vehicle_tracks = set()
        if getattr(self, '_db_cleanup', None) is not None:
            self._db_cleanup.stop()

        self._set_live_badge(False)
        self._sync_controls_for_state()
        self.statusBar.showMessage(self._tr.tr("status_stopped"))

    def _sync_controls_for_state(self):
        """根据处理状态同步控件启用/禁用（避免误操作）"""
        running = bool(self.video_thread and self.video_thread.isRunning()) or self._is_processing
        # 打开视频/摄像头始终可用（内部会先停止旧线程再切换）
        self.btn_open.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.btn_stop.setEnabled(running)
        if running and self._is_paused:
            self.btn_stop.setText(self._tr.tr("btn_resume"))
        elif running:
            self.btn_stop.setText(self._tr.tr("btn_pause"))
        else:
            self.btn_stop.setText(self._tr.tr("btn_stop"))

    def _on_frame_ready(self, frame: np.ndarray, tracks: list,
                        features_list: list, violations: list,
                        collision_risks: list = None, plate_results: dict = None):
        """Process frame"""
        self.current_frame = frame.copy()

        # 保存车牌识别结果
        if plate_results:
            self._plate_cache.update(plate_results)

        for i, track in enumerate(tracks):
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 构建标签
            if i < len(features_list):
                f = features_list[i]
                label = f"ID:{track.track_id} {f.color} {f.speed:.1f}km/h"
            else:
                label = f"ID:{track.track_id} {track.class_name}"

            # 添加车牌信息
            plate = getattr(self, '_plate_cache', {}).get(track.track_id)
            if plate:
                # OpenCV 不支持中文省份简称，这里仅显示字母数字部分，避免出现“????”
                ascii_plate = re.sub(r'[^A-Z0-9]', '', plate.upper())
                if ascii_plate:
                    label += f" [{ascii_plate}]"
                    # 在车辆下方显示字母数字部分
                    cv2.putText(frame, ascii_plate, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        self._display_frame(frame)
        self.card_vehicles.set_value(str(len(tracks)))

        if features_list:
            avg_speed = sum(f.speed for f in features_list) / len(features_list)
            self.card_speed.set_value(f"{avg_speed:.1f} km/h")

        self._last_collision_risks = collision_risks or []
        self._refresh_risk_labels()

        self.vehicle_table.setRowCount(len(tracks))
        for i, track in enumerate(tracks):
            self.vehicle_table.setItem(i, 0, QTableWidgetItem(str(track.track_id)))
            self.vehicle_table.setItem(i, 1, QTableWidgetItem(track.class_name))

            if i < len(features_list):
                f = features_list[i]
                self.vehicle_table.setItem(i, 2, QTableWidgetItem(f.color))
                self.vehicle_table.setItem(i, 3, QTableWidgetItem(f"{f.speed:.1f}"))
                self.vehicle_table.setItem(i, 4, QTableWidgetItem(f.direction.value))
            else:
                self.vehicle_table.setItem(i, 2, QTableWidgetItem("-"))
                self.vehicle_table.setItem(i, 3, QTableWidgetItem("-"))
                self.vehicle_table.setItem(i, 4, QTableWidgetItem("-"))

            # 添加车牌信息
            plate = self._plate_cache.get(track.track_id, "-")
            self.vehicle_table.setItem(i, 5, QTableWidgetItem(plate if plate else "-"))

            # 维护车辆数据库记录（支持车牌检索）
            if i < len(features_list):
                f = features_list[i]
                clean_plate = plate if plate and not self._tr.is_plate_pending(plate) and plate != "-" else None
                if track.track_id in self._seen_vehicle_tracks:
                    self.database.update_vehicle(
                        track_id=track.track_id,
                        speed=f.speed,
                        direction=f.direction.value,
                        plate_number=clean_plate,
                        vehicle_type=track.class_name,
                        color=f.color,
                    )
                else:
                    self.database.add_vehicle(
                        track_id=track.track_id,
                        plate_number=clean_plate,
                        vehicle_type=track.class_name,
                        color=f.color,
                        speed=f.speed,
                        direction=f.direction.value,
                    )
                    self._seen_vehicle_tracks.add(track.track_id)

        for record in violations:
            self._add_violation_record(record)

        # 当前帧有新增违规时，刷新一次（让过滤状态始终正确）
        if violations:
            self._refresh_violation_table()

        # Update statistics charts (every 30 frames to reduce overhead)
        if hasattr(self, '_frame_counter'):
            self._frame_counter += 1
        else:
            self._frame_counter = 0

        if self._frame_counter % 30 == 0:
            speeds = [f.speed for f in features_list if f.speed > 0]
            violation_dict = {}
            for v in violations:
                vtype = v.violation_type.value
                violation_dict[vtype] = violation_dict.get(vtype, 0) + 1

            vehicle_types = {}
            for track in tracks:
                vtype = track.class_name
                vehicle_types[vtype] = vehicle_types.get(vtype, 0) + 1

            # 更新右侧统计图
            self.stats_canvas.update_data(
                vehicle_count=len(tracks),
                speeds=speeds,
                violations=violation_dict,
                vehicle_types=vehicle_types
            )

            # 将实时流量写入数据库的 traffic_flow 表，便于后续分析
            if tracks:
                avg_speed = sum(f.speed for f in features_list) / max(1, len(features_list))
                # 使用出现最多的方向作为总体方向
                dir_counts = {}
                for f in features_list:
                    dir_counts[f.direction.value] = dir_counts.get(f.direction.value, 0) + 1
                dominant_dir = max(dir_counts, key=dir_counts.get) if dir_counts else "unknown"
                try:
                    self.database.add_traffic_flow(
                        vehicle_count=len(tracks),
                        avg_speed=avg_speed,
                        direction=dominant_dir,
                    )
                except Exception:
                    # GUI 下写流量失败不影响主流程，静默忽略
                    pass

    def _on_stats_updated(self, stats: dict):
        """Update statistics"""
        self._last_stats = stats
        tr = self._tr
        perf = stats.get('performance') or {}
        if perf:
            fps_val = perf.get('avg_fps', '--')
            self.card_fps.set_value(f"{fps_val}" if fps_val != '--' else "--")
            self.label_perf_degrade_value.setText(
                tr.tr(
                    "perf_line",
                    imgsz=perf.get('imgsz', '--'),
                    level=perf.get('degradation_level', 0),
                    skip=perf.get('frame_skip', 1),
                )
            )
        self.card_emergency.set_value(str(stats.get('emergency_vehicles', 0)))
        self._refresh_violation_labels()

        risk_stats = stats.get('collision_risks', {})
        if risk_stats:
            total_risks = risk_stats.get('total_risks', 0)
            critical = risk_stats.get('critical', 0)
            high = risk_stats.get('high', 0)
            self.label_risk_count.setText(
                tr.tr("active_risks_detail", total=total_risks, critical=critical, high=high)
            )

    def _display_frame(self, frame: np.ndarray):
        """Display frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled = qt_image.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def _add_violation_record(self, record: ViolationRecord):
        """缓存违规记录并写入数据库（与 CLI 行为一致）。"""
        self._violation_cache.append(record)
        plate = record.plate_number
        if plate in (None, '', '-') or self._tr.is_plate_pending(plate):
            plate = None
        try:
            self.database.add_violation(
                track_id=record.track_id,
                violation_type=record.violation_type.value,
                location=record.location,
                speed=record.speed,
                plate_number=plate,
                snapshot_path=record.snapshot_path,
                record_id=record.record_id,
                is_exempted=record.is_anomaly,
                exemption_reason=(
                    record.anomaly_reason.value if record.is_anomaly else None
                ),
                exemption_details=(
                    ", ".join(record.nearby_objects) if record.nearby_objects else None
                ),
                nearby_emergency_vehicles=record.nearby_objects,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "Failed to persist violation record %s", record.record_id,
            )

    def _refresh_violation_table(self):
        """根据过滤条件重建违规表（切换过滤即时生效，避免越跑越卡）"""
        if not hasattr(self, "violation_table"):
            return

        only_exempted = self.cb_only_exempted.isChecked()
        show_exempted = self.cb_show_exempted.isChecked()

        # 从缓存取最近 N 条展示
        rows: List[ViolationRecord] = []
        for rec in reversed(self._violation_cache):
            if only_exempted and not rec.is_anomaly:
                continue
            if not show_exempted and rec.is_anomaly:
                continue
            rows.append(rec)
            if len(rows) >= self._max_violation_rows:
                break
        rows.reverse()

        self.violation_table.setUpdatesEnabled(False)
        self.violation_table.setRowCount(0)
        for record in rows:
            row = self.violation_table.rowCount()
            self.violation_table.insertRow(row)

            self.violation_table.setItem(
                row, 0, QTableWidgetItem(record.timestamp.strftime("%H:%M:%S"))
            )
            self.violation_table.setItem(
                row, 1, QTableWidgetItem(record.violation_type.value)
            )
            self.violation_table.setItem(
                row, 2, QTableWidgetItem(record.plate_number or "-")
            )

            speed_str = f"{record.speed:.1f}" if record.speed else "-"
            self.violation_table.setItem(row, 3, QTableWidgetItem(speed_str))

            status_key = "status_review" if record.is_anomaly else "status_violation"
            status_item = QTableWidgetItem(self._tr.tr(status_key))
            if record.is_anomaly:
                status_item.setBackground(QColor(253, 230, 138, 80))  # 柔和黄
                status_item.setForeground(QColor("#e5e7eb"))
            else:
                status_item.setBackground(QColor(252, 165, 165, 70))  # 柔和红
                status_item.setForeground(QColor("#e5e7eb"))
            self.violation_table.setItem(row, 4, status_item)

            reason = ""
            if record.is_anomaly:
                reason_map = {
                    AnomalyReason.EMERGENCY_VEHICLE: "Emergency nearby",
                    AnomalyReason.TRAFFIC_POLICE: "Police directing",
                    AnomalyReason.SIGNAL_MALFUNCTION: "Signal abnormal",
                    AnomalyReason.NONE: "",
                }
                reason = reason_map.get(record.anomaly_reason, "")
            self.violation_table.setItem(row, 5, QTableWidgetItem(reason))

            details = (
                ", ".join(record.nearby_objects)
                if record.is_anomaly and record.nearby_objects
                else f"Location: {record.location}"
            )
            self.violation_table.setItem(row, 6, QTableWidgetItem(details))

        self.violation_table.setUpdatesEnabled(True)

    def _refresh_db_table(self):
        """刷新数据库查看表（traffic_flow / vehicles / violations）"""
        if not hasattr(self, "db_table"):
            return
        table = self.db_table_combo.currentText()
        plate_filter = self.db_search_input.text().strip()

        try:
            rows = self.database.get_table(table, limit=200)
        except Exception as exc:
            QMessageBox.critical(self, self._tr.tr("db_error_title"), str(exc))
            return

        # 按车牌过滤（仅 vehicles / violations）
        if plate_filter and table in ("vehicles", "violations"):
            key = "plate_number"
            rows = [r for r in rows if plate_filter in str(r.get(key, "") or "")]

        # 更新删除按钮状态（仅 vehicles 支持删除）
        self.db_delete_btn.setEnabled(table == "vehicles")

        self.db_table.setUpdatesEnabled(False)
        self.db_table.setRowCount(0)

        if not rows:
            self.db_table.setColumnCount(0)
            self.db_table.setHorizontalHeaderLabels([])
            self.db_table.setUpdatesEnabled(True)
            return

        # 根据首行动态生成列
        keys = list(rows[0].keys())
        self.db_table.setColumnCount(len(keys))
        self.db_table.setHorizontalHeaderLabels(keys)

        for item in rows:
            row = self.db_table.rowCount()
            self.db_table.insertRow(row)
            for col, key in enumerate(keys):
                val = item.get(key)
                if val is None:
                    text = ""
                elif isinstance(val, float):
                    text = f"{val:.2f}"
                else:
                    text = str(val)
                self.db_table.setItem(row, col, QTableWidgetItem(text))

        self.db_table.setUpdatesEnabled(True)

    def _delete_selected_db_vehicles(self):
        """删除数据库中选中的车辆记录"""
        if not hasattr(self, "db_table"):
            return
        if self.db_table_combo.currentText() != "vehicles":
            QMessageBox.information(
                self, self._tr.tr("db_delete_title"), self._tr.tr("db_delete_only_vehicles"),
            )
            return
        rows = self.db_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(
                self, self._tr.tr("db_delete_title"), self._tr.tr("db_delete_select"),
            )
            return
        ids = []
        for index in rows:
            id_item = self.db_table.item(index.row(), 0)
            if id_item:
                try:
                    ids.append(int(id_item.text()))
                except ValueError:
                    continue
        if not ids:
            return

        box = QMessageBox(self)
        box.setWindowTitle(self._tr.tr("db_confirm_delete_title"))
        box.setText(self._tr.tr("db_confirm_delete", n=len(ids)))
        box.setIcon(QMessageBox.Warning)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.setStyleSheet(message_box_stylesheet())
        reply = box.exec_()
        if reply != QMessageBox.Yes:
            return

        try:
            deleted = self.database.delete_vehicles_by_ids(ids)
        except Exception as exc:
            QMessageBox.critical(self, self._tr.tr("db_error_title"), str(exc))
            return

        # 美化后的“删除完成”提示框
        info_box = QMessageBox(self)
        info_box.setWindowTitle(self._tr.tr("db_deleted_title"))
        info_box.setText(self._tr.tr("db_deleted", n=deleted))
        info_box.setIcon(QMessageBox.Information)
        info_box.setStandardButtons(QMessageBox.Ok)
        info_box.setDefaultButton(QMessageBox.Ok)
        info_box.setStyleSheet(message_box_stylesheet())
        info_box.exec_()
        self._refresh_db_table()

    def _clean_old_db_records(self):
        """清空当前所选数据库表的所有记录"""
        table = self.db_table_combo.currentText()
        box = QMessageBox(self)
        box.setWindowTitle(self._tr.tr("db_confirm_clean_title"))
        box.setText(self._tr.tr("db_confirm_clean_table", table=table))
        box.setIcon(QMessageBox.Question)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.setStyleSheet(message_box_stylesheet())
        reply = box.exec_()
        if reply != QMessageBox.Yes:
            return
        try:
            deleted = self.database.clear_table(table)
        except Exception as exc:
            QMessageBox.critical(self, self._tr.tr("db_error_title"), str(exc))
            return
        QMessageBox.information(
            self, self._tr.tr("clean_complete_title"),
            self._tr.tr("clean_done_table", table=table, count=deleted),
        )
        self._refresh_db_table()

    def _search_plate(self):
        """Search plate"""
        plate = self.search_input.text().strip()
        if not plate:
            return

        results = self.database.search_by_plate(plate)
        if results:
            QMessageBox.information(
                self,
                self._tr.tr("btn_search"),
                self._tr.tr("search_found", n=len(results)),
            )
        else:
            QMessageBox.information(
                self, self._tr.tr("btn_search"), self._tr.tr("search_not_found"),
            )

    def _reset_statistics(self):
        """Reset statistics charts"""
        self.stats_canvas.reset()
        self._frame_counter = 0
        self.statusBar.showMessage(self._tr.tr("status_stats_reset"))

    def _on_error(self, error: str):
        """Handle error"""
        self._is_processing = False
        self._sync_controls_for_state()
        QMessageBox.critical(self, self._tr.tr("error_title"), error)
        self._stop_video()

    def closeEvent(self, event):
        """Close event"""
        self._stop_video()
        event.accept()


def main():
    """Main function"""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
