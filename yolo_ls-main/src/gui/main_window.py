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
    QCheckBox, QProgressBar, QFrame, QGridLayout
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
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.video import VideoStream
from src.core import VehicleDetector, ByteTracker, FeatureExtractor
from src.core.adaptive_violation import AdaptiveViolationDetector, ViolationRecord, AnomalyReason
from src.core.stgat import VehicleInteractionGraph
from src.core.collision_risk import CollisionRiskPredictor, RiskLevel
from src.ocr import PlateReader
from src.ocr.ocr_scheduler import OCRScheduler
from src.database import Database
from src.database.scheduler import start_db_cleanup_from_config
from src.utils.config import load_config
from src.utils.model_manager import ModelManager
from src.utils.performance import PerformanceOptimizer, FPSMonitor


class StatisticsCanvas(FigureCanvas):
    """Matplotlib canvas for statistics charts"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor='#1a1a2e')
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

    def _setup_style(self):
        """Setup chart style"""
        for ax in [self.ax_flow, self.ax_violation, self.ax_speed, self.ax_type]:
            ax.set_facecolor('#2d2d44')
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#4a4a6a')
            ax.title.set_color('white')

        self.ax_flow.set_title('Traffic Flow', fontsize=10)
        self.ax_violation.set_title('Violation Types', fontsize=10)
        self.ax_speed.set_title('Speed Distribution', fontsize=10)
        self.ax_type.set_title('Vehicle Types', fontsize=10)

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
        self._setup_ax(self.ax_flow, 'Traffic Flow')
        if self.vehicle_count_data:
            x = list(range(len(self.vehicle_count_data)))
            self.ax_flow.plot(x, list(self.vehicle_count_data), color='#00ff88', linewidth=2)
            self.ax_flow.fill_between(x, list(self.vehicle_count_data), alpha=0.3, color='#00ff88')
            self.ax_flow.set_ylabel('Count', fontsize=8, color='white')

        # Violation pie
        self.ax_violation.clear()
        self._setup_ax(self.ax_violation, 'Violation Types')
        if self.violation_counts:
            labels = list(self.violation_counts.keys())
            sizes = list(self.violation_counts.values())
            colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff'][:len(labels)]
            self.ax_violation.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                                  textprops={'color': 'white', 'fontsize': 7})
        else:
            self.ax_violation.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray', fontsize=10)

        # Speed histogram
        self.ax_speed.clear()
        self._setup_ax(self.ax_speed, 'Speed Distribution')
        if self.speed_data:
            self.ax_speed.hist(list(self.speed_data), bins=15, color='#4d96ff', alpha=0.7, edgecolor='white')
            self.ax_speed.axvline(x=60, color='#ff6b6b', linestyle='--', linewidth=1.5)
            self.ax_speed.set_xlabel('km/h', fontsize=8, color='white')
        else:
            self.ax_speed.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray', fontsize=10)

        # Vehicle type bar
        self.ax_type.clear()
        self._setup_ax(self.ax_type, 'Vehicle Types')
        if self.vehicle_type_counts:
            types = list(self.vehicle_type_counts.keys())
            counts = list(self.vehicle_type_counts.values())
            colors = ['#6bcb77', '#4d96ff', '#ffd93d', '#ff6b6b'][:len(types)]
            self.ax_type.bar(types, counts, color=colors)
            self.ax_type.tick_params(axis='x', labelrotation=15)
        else:
            self.ax_type.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray', fontsize=10)

        self.fig.tight_layout(pad=2.0)
        self.draw()

    def _setup_ax(self, ax, title: str):
        """Setup axis style"""
        ax.set_facecolor('#2d2d44')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#4a4a6a')
        ax.set_title(title, fontsize=10, color='white')

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

        self.video_stream: Optional[VideoStream] = None
        self.detector: Optional[VehicleDetector] = None
        self.tracker: Optional[ByteTracker] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.violation_detector: Optional[AdaptiveViolationDetector] = None
        self.interaction_graph: Optional[VehicleInteractionGraph] = None
        self.collision_predictor: Optional[CollisionRiskPredictor] = None
        self.plate_reader: Optional[PlateReader] = None

    def run(self):
        """Run video processing"""
        try:
            self.video_stream = VideoStream(
                self.source,
                fps=self.config.get('fps', 15)
            )
            if not self.video_stream.open():
                self.error.emit("Cannot open video source")
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

            self.detector = VehicleDetector(
                model_path=yolo_path,
                confidence=self.config.get('confidence', 0.5),
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

            self.violation_detector = AdaptiveViolationDetector(
                speed_limit=self.config.get('speed_limit', 60),
                snapshot_dir=self.config.get('snapshot_dir', 'data/snapshots'),
                emergency_distance=self.config.get('emergency_distance', 300),
                wrong_way_enabled=self.config.get('wrong_way_enabled', True),
                illegal_lane_enabled=self.config.get('illegal_lane_enabled', True),
                expected_flow_direction=self.config.get('expected_flow_direction', 'south'),
                lane_change_lateral_px=self.config.get('lane_change_lateral_px', 80),
                lane_change_min_speed_kmh=self.config.get('lane_change_min_speed_kmh', 15),
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

            # Initialize Plate Reader（可配置关闭以减轻 Paddle 加载）
            if self.config.get('ocr_enabled', True) and int(self.config.get('ocr_interval', 0)) > 0:
                self.plate_reader = PlateReader(
                    model_path=self.config.get('plate_model_path', 'models/plate_ocr.pt'),
                    use_gpu=self.config.get('device', 'cpu') != 'cpu',
                    paddle_mobile=self.config.get('ocr_paddle_mobile', True),
                )
            else:
                self.plate_reader = None

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

            for frame in self.video_stream.frames():
                if not self.running:
                    break

                frame_start = time.perf_counter()
                frame_count += 1

                if perf_enabled and not perf.should_process_frame(frame_count):
                    continue

                if perf_enabled:
                    self.detector.imgsz = perf.get_imgsz()
                    self.detector.enable_tiling = perf.get_enable_tiling()

                effective_risk_interval = (
                    perf.get_risk_interval(base_risk_interval) if perf_enabled else base_risk_interval
                )
                effective_ocr_interval = (
                    perf.get_ocr_interval(base_ocr_interval) if perf_enabled else base_ocr_interval
                )
                risk_active = bool(self.config.get('enable_risk', True)) and not (
                    perf_enabled and perf.risk_disabled()
                )

                # ===== 1) 检测：单次推理为主，避免每帧多次 predict 导致卡顿 =====
                enable_context = bool(self.config.get('enable_context_detection', False))
                if enable_context:
                    # 车辆 + 人 + 信号灯一次性检测（比三次 predict 快很多）
                    all_classes = list(self.detector.VEHICLE_CLASSES.keys()) + [
                        self.detector.PERSON_CLASS,
                        self.detector.TRAFFIC_LIGHT_CLASS,
                    ]
                    all_dets = self.detector.detect(frame, all_classes)
                    vehicle_dets = [d for d in all_dets if d.class_id in self.detector.VEHICLE_CLASSES]
                    person_bboxes = [d.bbox for d in all_dets if d.class_id == self.detector.PERSON_CLASS]
                    light_dets = [d for d in all_dets if d.class_id == self.detector.TRAFFIC_LIGHT_CLASS]
                    light_bbox = max(light_dets, key=lambda d: d.confidence).bbox if light_dets else None
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
                if self.plate_reader and effective_ocr_interval > 0 and frame_count % effective_ocr_interval == 0:
                    bbox_heights = {t.track_id: t.bbox[3] - t.bbox[1] for t in tracks}
                    ocr_targets = ocr_scheduler.select_tracks(
                        [t.track_id for t in tracks],
                        plate_cache,
                        bbox_heights,
                        ocr_min_h,
                    )

                for track in tracks:
                    plate_number = plate_cache.get(track.track_id)
                    if self.plate_reader and track.track_id in ocr_targets:
                        plate_result = self.plate_reader.read(frame, track.bbox)
                        if plate_result:
                            plate_number = plate_result.plate_number
                            plate_cache[track.track_id] = plate_number
                            plate_results[track.track_id] = plate_number
                        elif track.track_id not in plate_cache:
                            plate_cache[track.track_id] = "识别中"
                            plate_results[track.track_id] = "识别中"
                    elif (
                        self.plate_reader
                        and plate_number
                        and plate_number != "识别中"
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

                if perf_enabled and fps_monitor and frame_count % 30 == 0:
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

                if frame_count % 30 == 0:
                    stats = self.violation_detector.get_statistics()
                    stats['emergency_vehicles'] = len(self.violation_detector.current_emergency_vehicles)
                    if risk_active and self.collision_predictor:
                        stats['collision_risks'] = self.collision_predictor.get_risk_summary(collision_risks)
                    else:
                        stats['collision_risks'] = {}
                    if perf_enabled and fps_monitor:
                        pstatus = perf.get_status()
                        stats['performance'] = {
                            'avg_fps': round(fps_monitor.avg_fps, 1),
                            'imgsz': pstatus['imgsz'],
                            'degradation_level': pstatus['degradation_level'],
                            'frame_skip': pstatus['frame_skip'],
                        }
                    self.stats_updated.emit(stats)

                if perf_enabled and fps_monitor:
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
            if self.video_stream:
                self.video_stream.release()

    def stop(self):
        """Stop processing"""
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    """Main Window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Traffic Analysis - Adaptive Violation Detection")
        self.setGeometry(100, 100, 1500, 950)

        # 全局外观：深色主题 + 统一字体
        self._apply_global_style()

        self.video_thread: Optional[VideoThread] = None
        self.current_frame: Optional[np.ndarray] = None
        self._plate_cache: Dict[int, str] = {}
        self._seen_vehicle_tracks = set()
        # 交互优化：缓存最近违规记录，便于过滤切换即时刷新
        self._violation_cache: deque[ViolationRecord] = deque(maxlen=500)
        self._max_violation_rows = 200
        self._is_processing = False

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
            'ocr_max_vehicles_per_frame': 10,
            'performance_warmup_frames': 45,
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
        }

        self._merge_config_from_yaml()
        self.database = Database(
            db_path=self.config.get('database_path', 'data/traffic.db'),
            pool_size=int(self.config.get('database_pool_size', 5)),
        )
        self._db_cleanup = start_db_cleanup_from_config(
            self.database,
            load_config('config/settings.yaml') or {'database': {}},
        )
        self._init_ui()
        gui_cfg = (load_config('config/settings.yaml') or {}).get('gui', {})
        title = gui_cfg.get('window_title')
        if title:
            self.setWindowTitle(title)
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
        })

    def _apply_config_to_controls(self):
        """将 config 同步到设置面板控件。"""
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

    def _apply_global_style(self):
        """应用全局深色主题和基础样式（不影响业务逻辑）"""
        palette = self.palette()
        # 更柔和、略微明亮的深色方案
        palette.setColor(palette.Window, QColor("#0f172a"))          # 全局背景
        palette.setColor(palette.Base, QColor("#020617"))            # 输入/表格底色
        palette.setColor(palette.AlternateBase, QColor("#020617"))
        palette.setColor(palette.Text, QColor("#e5e7eb"))
        palette.setColor(palette.WindowText, QColor("#e5e7eb"))
        palette.setColor(palette.Button, QColor("#0b1120"))
        palette.setColor(palette.ButtonText, QColor("#e5e7eb"))
        palette.setColor(palette.Highlight, QColor("#38bdf8"))  # 清爽蓝
        palette.setColor(palette.HighlightedText, QColor("#0f172a"))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
            }

            QWidget {
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 12px;
            }

            QLabel {
                color: #e5e7eb;
            }

            QGroupBox {
                color: #e5e7eb;
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 12px;
                margin-top: 10px;
                padding: 10px 12px 14px 12px;
                font-weight: 600;
                background-color: rgba(15, 23, 42, 0.55);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #94a3b8;
            }

            QTabWidget::pane {
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 10px;
                background-color: #020617;
            }
            QTabWidget {
                background: transparent;
            }
            QTabBar::tab {
                background: #0b1220;
                color: #94a3b8;
                padding: 7px 14px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #0f172a;
                color: #e5e7eb;
            }
            QTabBar::tab:hover {
                background: rgba(148, 163, 184, 0.10);
            }

            QSplitter::handle {
                background: rgba(148, 163, 184, 0.08);
            }
            QSplitter::handle:hover {
                background: rgba(96, 165, 250, 0.18);
            }

            QTableWidget {
                background-color: #020617;
                alternate-background-color: rgba(148, 163, 184, 0.07);
                gridline-color: rgba(148, 163, 184, 0.12);
                color: #e5e7eb;
                selection-background-color: rgba(96, 165, 250, 0.35);
                selection-color: #f9fafb;
                border-radius: 8px;
            }
            QTableCornerButton::section {
                background-color: #0b1220;
                border: none;
            }
            QHeaderView::section {
                background-color: #020617;
                color: #94a3b8;
                padding: 6px 6px;
                border: none;
                border-bottom: 1px solid rgba(148, 163, 184, 0.12);
            }

            QLineEdit {
                background-color: #020617;
                border-radius: 8px;
                border: 1px solid rgba(148, 163, 184, 0.16);
                padding: 6px 10px;
                color: #e5e7eb;
            }
            QLineEdit:focus {
                border-color: rgba(96, 165, 250, 0.70);
            }

            QCheckBox {
                color: #e5e7eb;
            }

            QPushButton {
                background-color: rgba(148, 163, 184, 0.10);
                color: #e5e7eb;
                border: 1px solid rgba(148, 163, 184, 0.10);
                border-radius: 16px;
                padding: 8px 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(148, 163, 184, 0.14);
                border-color: rgba(148, 163, 184, 0.16);
            }
            QPushButton:pressed {
                background-color: rgba(96, 165, 250, 0.18);
                border-color: rgba(96, 165, 250, 0.22);
            }
            QPushButton:disabled {
                background-color: rgba(148, 163, 184, 0.06);
                color: rgba(148, 163, 184, 0.45);
                border-color: rgba(148, 163, 184, 0.06);
            }

            QSpinBox {
                background-color: #020617;
                border-radius: 8px;
                border: 1px solid rgba(148, 163, 184, 0.16);
                padding: 4px 10px;
                color: #e5e7eb;
            }
            QSpinBox:focus {
                border-color: rgba(96, 165, 250, 0.70);
            }

            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 6px 2px 6px 2px;
            }
            QScrollBar::handle:vertical {
                background: rgba(148, 163, 184, 0.20);
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(148, 163, 184, 0.30);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }

            QScrollBar:horizontal {
                background: transparent;
                height: 10px;
                margin: 2px 6px 2px 6px;
            }
            QScrollBar::handle:horizontal {
                background: rgba(148, 163, 184, 0.20);
                border-radius: 5px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: rgba(148, 163, 184, 0.30);
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: transparent;
            }

            QStatusBar {
                background-color: #020617;
                color: #94a3b8;
            }
        """)

    def _init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        # Left panel: Video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(900, 650)
        self.video_label.setStyleSheet("""
            background-color: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #0b1220,
                stop:1 #0f172a
            );
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.14);
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("拖拽视频到此处，或点击下方“打开视频”")
        self.video_label.setStyleSheet(self.video_label.styleSheet() + "color: #94a3b8; font-size: 13px;")
        left_layout.addWidget(self.video_label)

        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        self.btn_open = QPushButton("Open Video")
        self.btn_camera = QPushButton("Open Camera")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        # 单独给底部按钮一个更清晰的层级
        for btn in [self.btn_open, self.btn_camera]:
            btn.setMinimumHeight(38)
            btn.setStyleSheet(btn.styleSheet() + "QPushButton{background-color: rgba(96, 165, 250, 0.16);} QPushButton:hover{background-color: rgba(96, 165, 250, 0.22);}")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setStyleSheet(self.btn_stop.styleSheet() + "QPushButton{background-color: rgba(252, 165, 165, 0.14);} QPushButton:hover{background-color: rgba(252, 165, 165, 0.20);}")

        self.btn_open.clicked.connect(self._open_video)
        self.btn_camera.clicked.connect(self._open_camera)
        self.btn_stop.clicked.connect(self._stop_video)

        btn_layout.addWidget(self.btn_open)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_stop)
        left_layout.addLayout(btn_layout)

        # Right panel: Info tabs
        right_panel = QTabWidget()
        right_panel.setMaximumWidth(550)
        right_panel.setDocumentMode(True)
        right_panel.setUsesScrollButtons(True)

        # ===== Real-time Info Tab =====
        info_tab = QWidget()
        info_layout = QVBoxLayout(info_tab)

        stats_group = QGroupBox("Real-time Statistics")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(6)

        # 使用两列栅格让右上角信息更整齐
        stats_header_grid = QGridLayout()
        stats_header_grid.setColumnStretch(0, 1)
        stats_header_grid.setColumnStretch(1, 1)

        self.label_vehicle_count = QLabel("Vehicles")
        self.label_emergency_count = QLabel("Emergency Vehicles")
        self.label_avg_speed = QLabel("Avg Speed")
        for lbl in (self.label_vehicle_count, self.label_emergency_count, self.label_avg_speed):
            lbl.setStyleSheet("color: #9ca3af;")

        self.label_vehicle_count_value = QLabel("0")
        self.label_emergency_count_value = QLabel("0")
        self.label_avg_speed_value = QLabel("0 km/h")
        for v in (self.label_vehicle_count_value, self.label_emergency_count_value, self.label_avg_speed_value):
            v.setAlignment(Qt.AlignRight)

        stats_header_grid.addWidget(self.label_vehicle_count, 0, 0)
        stats_header_grid.addWidget(self.label_vehicle_count_value, 0, 1)
        stats_header_grid.addWidget(self.label_emergency_count, 1, 0)
        stats_header_grid.addWidget(self.label_emergency_count_value, 1, 1)
        stats_header_grid.addWidget(self.label_avg_speed, 2, 0)
        stats_header_grid.addWidget(self.label_avg_speed_value, 2, 1)

        self.label_perf_fps = QLabel("Runtime FPS")
        self.label_perf_fps.setStyleSheet("color: #9ca3af;")
        self.label_perf_fps_value = QLabel("--")
        self.label_perf_fps_value.setAlignment(Qt.AlignRight)
        self.label_perf_degrade = QLabel("Perf / Degrade")
        self.label_perf_degrade.setStyleSheet("color: #9ca3af;")
        self.label_perf_degrade_value = QLabel("imgsz -- L0")
        self.label_perf_degrade_value.setAlignment(Qt.AlignRight)
        stats_header_grid.addWidget(self.label_perf_fps, 3, 0)
        stats_header_grid.addWidget(self.label_perf_fps_value, 3, 1)
        stats_header_grid.addWidget(self.label_perf_degrade, 4, 0)
        stats_header_grid.addWidget(self.label_perf_degrade_value, 4, 1)

        violation_frame = QFrame()
        violation_frame.setStyleSheet("background-color: rgba(15, 23, 42, 0.35); border-radius: 10px; padding: 10px; border: 1px solid rgba(148, 163, 184, 0.12);")
        vf_layout = QVBoxLayout(violation_frame)
        self.label_total_violations = QLabel("Total Violations: 0")
        self.label_actual_violations = QLabel("Actual Violations: 0")
        self.label_actual_violations.setStyleSheet("color: #fca5a5; font-weight: 700;")
        self.label_exempted = QLabel("Exempted (Special Cases): 0")
        self.label_exempted.setStyleSheet("color: #fde68a; font-weight: 700;")
        vf_layout.addWidget(self.label_total_violations)
        vf_layout.addWidget(self.label_actual_violations)
        vf_layout.addWidget(self.label_exempted)

        stats_layout.addLayout(stats_header_grid)
        stats_layout.addWidget(violation_frame)

        # Collision Risk Display
        risk_frame = QFrame()
        risk_frame.setStyleSheet("background-color: rgba(15, 23, 42, 0.35); border-radius: 10px; padding: 10px; border: 1px solid rgba(148, 163, 184, 0.12);")
        rf_layout = QVBoxLayout(risk_frame)
        self.label_collision_risk = QLabel("Collision Risk: SAFE")
        self.label_collision_risk.setStyleSheet("color: #86efac; font-weight: 700;")
        self.label_min_ttc = QLabel("Min TTC: --")
        self.label_risk_count = QLabel("Active Risks: 0")
        rf_layout.addWidget(self.label_collision_risk)
        rf_layout.addWidget(self.label_min_ttc)
        rf_layout.addWidget(self.label_risk_count)
        stats_layout.addWidget(risk_frame)

        info_layout.addWidget(stats_group)

        # Vehicle list
        vehicle_group = QGroupBox("Detected Vehicles")
        vehicle_layout = QVBoxLayout(vehicle_group)
        self.vehicle_table = QTableWidget()
        self.vehicle_table.setColumnCount(6)
        self.vehicle_table.setHorizontalHeaderLabels(
            ["ID", "Type", "Color", "Speed(km/h)", "Direction", "Plate"]
        )
        self.vehicle_table.horizontalHeader().setStretchLastSection(True)
        self.vehicle_table.verticalHeader().setVisible(False)
        self.vehicle_table.setShowGrid(False)
        self.vehicle_table.setAlternatingRowColors(True)
        self.vehicle_table.setSelectionBehavior(self.vehicle_table.SelectRows)
        self.vehicle_table.setEditTriggers(self.vehicle_table.NoEditTriggers)
        vehicle_layout.addWidget(self.vehicle_table)
        info_layout.addWidget(vehicle_group)

        right_panel.addTab(info_tab, "Real-time Info")

        # ===== Violation Records Tab =====
        violation_tab = QWidget()
        violation_layout = QVBoxLayout(violation_tab)

        filter_layout = QHBoxLayout()
        self.cb_show_exempted = QCheckBox("Show Exempted")
        self.cb_show_exempted.setChecked(True)
        self.cb_only_exempted = QCheckBox("Only Exempted")
        filter_layout.addWidget(self.cb_show_exempted)
        filter_layout.addWidget(self.cb_only_exempted)
        filter_layout.addStretch()
        violation_layout.addLayout(filter_layout)

        self.violation_table = QTableWidget()
        self.violation_table.setColumnCount(7)
        self.violation_table.setHorizontalHeaderLabels(
            ["Time", "Type", "Plate", "Speed", "Status", "Reason", "Details"]
        )
        self.violation_table.horizontalHeader().setStretchLastSection(True)
        self.violation_table.setAlternatingRowColors(True)
        self.violation_table.verticalHeader().setVisible(False)
        self.violation_table.setShowGrid(False)
        self.violation_table.setSelectionBehavior(self.violation_table.SelectRows)
        self.violation_table.setEditTriggers(self.violation_table.NoEditTriggers)
        violation_layout.addWidget(self.violation_table)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter plate number...")
        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self._search_plate)
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(btn_search)
        violation_layout.addLayout(search_layout)
        # 交互优化：切换过滤立即刷新
        self.cb_show_exempted.stateChanged.connect(self._refresh_violation_table)
        self.cb_only_exempted.stateChanged.connect(self._refresh_violation_table)

        right_panel.addTab(violation_tab, "Violations")

        # ===== Vehicles DB Tab =====
        vehicles_tab = QWidget()
        vehicles_layout = QVBoxLayout(vehicles_tab)

        toolbar_layout = QHBoxLayout()
        self.db_table_combo = QComboBox()
        self.db_table_combo.addItems(["traffic_flow", "vehicles", "violations"])
        self.db_search_input = QLineEdit()
        self.db_search_input.setPlaceholderText("按车牌筛选（仅 vehicles/violations 生效）")
        self.db_refresh_btn = QPushButton("刷新")
        self.db_delete_btn = QPushButton("删除选中车辆")
        toolbar_layout.addWidget(QLabel("Table:"))
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

        # 底部：按天清理旧记录
        clean_layout = QHBoxLayout()
        clean_layout.addWidget(QLabel("清理早于 N 天的记录:"))
        self.db_clean_days_spin = QSpinBox()
        self.db_clean_days_spin.setRange(1, 365)
        self.db_clean_days_spin.setValue(30)
        self.db_clean_btn = QPushButton("清理旧记录")
        clean_layout.addWidget(self.db_clean_days_spin)
        clean_layout.addWidget(self.db_clean_btn)
        clean_layout.addStretch()
        vehicles_layout.addLayout(clean_layout)

        # 绑定事件
        self.db_refresh_btn.clicked.connect(self._refresh_db_table)
        self.db_delete_btn.clicked.connect(self._delete_selected_db_vehicles)
        self.db_clean_btn.clicked.connect(self._clean_old_db_records)
        self.db_table_combo.currentIndexChanged.connect(self._refresh_db_table)

        right_panel.addTab(vehicles_tab, "Database")

        # ===== Special Cases Info Tab =====
        exemption_tab = QWidget()
        exemption_layout = QVBoxLayout(exemption_tab)

        info_text = QLabel("""
<h3>Adaptive Violation Detection - Special Cases</h3>
<p>This system intelligently identifies special situations and marks them as exempted:</p>

<h4>1. Yielding to Emergency Vehicles</h4>
<p style="color: #ffd93d;">When ambulance, fire truck, or police car is detected nearby,
violations (running red light, lane crossing) are marked as "Yielding to Emergency".</p>

<h4>2. Traffic Light Malfunction</h4>
<p style="color: #ffd93d;">When traffic light shows abnormal status (no signal or irregular flashing),
related violations are marked as "Signal Malfunction".</p>

<h4>3. Other Special Cases</h4>
<ul>
<li>Police Direction</li>
<li>Emergency Avoidance</li>
<li>Road Construction Detour</li>
</ul>

<p><b>Note:</b> All special cases are recorded with snapshots.
Snapshot filenames include timestamp for later manual review.</p>
        """)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("padding: 10px;")
        exemption_layout.addWidget(info_text)
        exemption_layout.addStretch()

        right_panel.addTab(exemption_tab, "Special Cases")

        # ===== Statistics Tab =====
        stats_tab = QWidget()
        stats_tab_layout = QVBoxLayout(stats_tab)
        self.stats_canvas = StatisticsCanvas(stats_tab)
        stats_tab_layout.addWidget(self.stats_canvas)

        btn_reset_stats = QPushButton("Reset Statistics")
        btn_reset_stats.clicked.connect(self._reset_statistics)
        stats_tab_layout.addWidget(btn_reset_stats)

        right_panel.addTab(stats_tab, "Statistics")

        # ===== Settings Tab =====
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        detect_group = QGroupBox("Detection Settings")
        detect_layout = QVBoxLayout(detect_group)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.spin_confidence = QSpinBox()
        self.spin_confidence.setRange(1, 100)
        self.spin_confidence.setValue(20)
        self.spin_confidence.setSuffix("%")
        conf_layout.addWidget(self.spin_confidence)
        detect_layout.addLayout(conf_layout)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed Limit:"))
        self.spin_speed_limit = QSpinBox()
        self.spin_speed_limit.setRange(1, 200)
        self.spin_speed_limit.setValue(60)
        self.spin_speed_limit.setSuffix(" km/h")
        speed_layout.addWidget(self.spin_speed_limit)
        detect_layout.addLayout(speed_layout)

        emergency_layout = QHBoxLayout()
        emergency_layout.addWidget(QLabel("Emergency Distance:"))
        self.spin_emergency_dist = QSpinBox()
        self.spin_emergency_dist.setRange(50, 1000)
        self.spin_emergency_dist.setValue(300)
        self.spin_emergency_dist.setSuffix(" px")
        emergency_layout.addWidget(self.spin_emergency_dist)
        detect_layout.addLayout(emergency_layout)

        flow_layout = QHBoxLayout()
        flow_layout.addWidget(QLabel("Legal Flow Direction:"))
        self.combo_flow_direction = QComboBox()
        self.combo_flow_direction.addItems([
            'north', 'south', 'east', 'west',
            'northeast', 'northwest', 'southeast', 'southwest',
        ])
        self.combo_flow_direction.setCurrentText('south')
        flow_layout.addWidget(self.combo_flow_direction)
        detect_layout.addLayout(flow_layout)

        self.cb_enable_tiling = QCheckBox("Enable tiling (slower, better for distant vehicles)")
        self.cb_enable_tiling.setChecked(False)
        detect_layout.addWidget(self.cb_enable_tiling)

        self.cb_wrong_way = QCheckBox("Enable wrong-way detection (逆行)")
        self.cb_wrong_way.setChecked(True)
        detect_layout.addWidget(self.cb_wrong_way)

        self.cb_illegal_lane = QCheckBox("Enable illegal lane-change detection (违规变道)")
        self.cb_illegal_lane.setChecked(True)
        detect_layout.addWidget(self.cb_illegal_lane)

        self.cb_performance = QCheckBox("Enable adaptive performance (FPS monitor & degradation)")
        self.cb_performance.setChecked(True)
        detect_layout.addWidget(self.cb_performance)

        btn_reload_cfg = QPushButton("Reload config/settings.yaml")
        btn_reload_cfg.clicked.connect(lambda: (self._merge_config_from_yaml(), self._apply_config_to_controls()))
        detect_layout.addWidget(btn_reload_cfg)

        settings_layout.addWidget(detect_group)

        stopline_group = QGroupBox("Stop Line Settings")
        stopline_layout = QVBoxLayout(stopline_group)

        self.cb_enable_stopline = QCheckBox("Enable Stop Line Detection")
        stopline_layout.addWidget(self.cb_enable_stopline)

        sl_y_layout = QHBoxLayout()
        sl_y_layout.addWidget(QLabel("Y Position:"))
        self.spin_sl_y = QSpinBox()
        self.spin_sl_y.setRange(0, 2000)
        self.spin_sl_y.setValue(400)
        sl_y_layout.addWidget(self.spin_sl_y)
        stopline_layout.addLayout(sl_y_layout)

        sl_x_layout = QHBoxLayout()
        sl_x_layout.addWidget(QLabel("X Range:"))
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
        self.btn_auto_stopline = QPushButton("Auto Detect Stop Line")
        self.btn_auto_stopline.clicked.connect(self._auto_detect_stop_line)
        stopline_layout.addWidget(self.btn_auto_stopline)

        settings_layout.addWidget(stopline_group)
        settings_layout.addStretch()

        right_panel.addTab(settings_tab, "Settings")

        # Add to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        main_layout.addWidget(splitter)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def _open_video(self):
        """Open video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
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

        self.config['confidence'] = self.spin_confidence.value() / 100
        self.config['speed_limit'] = self.spin_speed_limit.value()
        self.config['emergency_distance'] = self.spin_emergency_dist.value()
        self.config['expected_flow_direction'] = self.combo_flow_direction.currentText()
        self.config['enable_tiling'] = self.cb_enable_tiling.isChecked()
        self.config['wrong_way_enabled'] = self.cb_wrong_way.isChecked()
        self.config['illegal_lane_enabled'] = self.cb_illegal_lane.isChecked()
        self.config['performance_enabled'] = self.cb_performance.isChecked()
        self.config['performance_target_fps'] = self.config.get('fps', 15)

        if self.cb_enable_stopline.isChecked():
            self.config['stop_line'] = {
                'y': self.spin_sl_y.value(),
                'x_start': self.spin_sl_x1.value(),
                'x_end': self.spin_sl_x2.value()
            }
        else:
            self.config['stop_line'] = None

        self._plate_cache = {}
        self._seen_vehicle_tracks = set()

        self.video_thread = VideoThread(source, self.config)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.stats_updated.connect(self._on_stats_updated)
        self.video_thread.error.connect(self._on_error)
        self.video_thread.start()

        self.statusBar.showMessage("Processing…  (press Stop to end)")

    def _auto_detect_stop_line(self):
        """
        基于当前画面自动估计一条停止线位置，并填入设置面板。
        实现思路：在下半部分 ROI 中用 Hough 检测近乎水平的亮线，聚合得到 y / x 范围。
        """
        def _styled_info(title: str, text: str):
            box = QMessageBox(self)
            box.setWindowTitle(title)
            box.setText(text)
            box.setIcon(QMessageBox.Information)
            box.setStandardButtons(QMessageBox.Ok)
            box.setDefaultButton(QMessageBox.Ok)
            box.setStyleSheet("""
                QMessageBox {
                    background-color: #020617;
                }
                QLabel {
                    color: #e5e7eb;
                    font-size: 12px;
                }
                QPushButton {
                    background-color: rgba(59, 130, 246, 0.22);
                    color: #dbeafe;
                    border-radius: 6px;
                    padding: 6px 16px;
                    min-width: 64px;
                }
                QPushButton:hover {
                    background-color: rgba(59, 130, 246, 0.30);
                }
            """)
            box.exec_()

        if self.current_frame is None:
            _styled_info("Auto Detect Stop Line", "当前没有视频帧可用，请先打开视频或摄像头。")
            return

        frame = self.current_frame.copy()
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            _styled_info("Auto Detect Stop Line", "当前帧尺寸无效，无法检测停止线。")
            return

        # 只在画面下半部分做检测，减少干扰
        roi_top = int(h * 0.45)
        roi = frame[roi_top:h, 0:w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 增强对亮色实线的响应
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

        # 形态学闭运算，连通断裂的线段
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=int(w * 0.35),
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            _styled_info("Auto Detect Stop Line", "未检测到明显的水平停车线，请在停止线附近暂停画面后重试，或手动设置。")
            return

        ys = []
        xs = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # 在 ROI 内的 y
            dy = y2 - y1
            dx = x2 - x1
            if dx == 0:
                continue
            slope = abs(dy / dx)
            # 只保留近乎水平的线段
            if slope > 0.15:
                continue
            y_mid = (y1 + y2) / 2.0
            ys.append(y_mid)
            xs.extend([x1, x2])

        if not ys or not xs:
            _styled_info("Auto Detect Stop Line", "未检测到稳定的停车线候选，请稍后再试或手动设置。")
            return

        # 采用中位数减少噪声影响
        y_roi = int(float(np.median(ys)))
        y_global = roi_top + y_roi
        x_start = max(0, int(min(xs)))
        x_end = min(w - 1, int(max(xs)))

        # 更新设置面板
        self.spin_sl_y.setValue(y_global)
        self.spin_sl_x1.setValue(x_start)
        self.spin_sl_x2.setValue(x_end)
        self.cb_enable_stopline.setChecked(True)

        # 尝试立即应用到运行中的检测线程
        try:
            if self.video_thread and self.video_thread.violation_detector:
                self.video_thread.violation_detector.set_stop_line(
                    y=y_global,
                    x_start=x_start,
                    x_end=x_end,
                )
        except Exception:
            # 失败不影响 GUI，其它地方会在下次 start 时应用
            pass

        # 友好提示
        _styled_info(
            "Auto Detect Stop Line",
            f"已自动检测停止线：Y = {y_global}, X 范围 [{x_start}, {x_end}]。\n"
            "你可以在 Settings 中微调后重新开始检测。",
        )

    def _stop_video(self):
        """Stop processing"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

        self._is_processing = False
        self._plate_cache = {}
        self._seen_vehicle_tracks = set()
        if getattr(self, '_db_cleanup', None) is not None:
            self._db_cleanup.stop()

        self._sync_controls_for_state()
        self.statusBar.showMessage("Stopped")

    def _sync_controls_for_state(self):
        """根据处理状态同步控件启用/禁用（避免误操作）"""
        running = bool(self.video_thread and self.video_thread.isRunning()) or self._is_processing
        self.btn_open.setEnabled(not running)
        self.btn_camera.setEnabled(not running)
        self.btn_stop.setEnabled(running)

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
        self.label_vehicle_count_value.setText(str(len(tracks)))

        if features_list:
            avg_speed = sum(f.speed for f in features_list) / len(features_list)
            self.label_avg_speed_value.setText(f"{avg_speed:.1f} km/h")

        # Update collision risk display
        if collision_risks:
            highest_risk = collision_risks[0] if collision_risks else None
            if highest_risk:
                risk_colors = {
                    RiskLevel.SAFE: "#86efac",
                    RiskLevel.LOW: "#fde68a",
                    RiskLevel.MEDIUM: "#fdba74",
                    RiskLevel.HIGH: "#fca5a5",
                    RiskLevel.CRITICAL: "#f0abfc"
                }
                color = risk_colors.get(highest_risk.risk_level, "#00ff00")
                self.label_collision_risk.setText(f"Collision Risk: {highest_risk.risk_level.value.upper()}")
                self.label_collision_risk.setStyleSheet(f"color: {color}; font-weight: bold;")

                if highest_risk.time_to_collision > 0:
                    self.label_min_ttc.setText(f"Min TTC: {highest_risk.time_to_collision:.1f}s")
                else:
                    self.label_min_ttc.setText("Min TTC: --")

                self.label_risk_count.setText(f"Active Risks: {len(collision_risks)}")
        else:
            self.label_collision_risk.setText("Collision Risk: SAFE")
            self.label_collision_risk.setStyleSheet("color: #00ff00; font-weight: bold;")
            self.label_min_ttc.setText("Min TTC: --")
            self.label_risk_count.setText("Active Risks: 0")

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
                clean_plate = plate if plate and plate not in ("-", "识别中") else None
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
        perf = stats.get('performance') or {}
        if perf:
            self.label_perf_fps_value.setText(f"{perf.get('avg_fps', '--')}")
            self.label_perf_degrade_value.setText(
                f"imgsz {perf.get('imgsz', '--')} · L{perf.get('degradation_level', 0)}"
                f" · skip {perf.get('frame_skip', 1)}"
            )
        self.label_emergency_count_value.setText(str(stats.get('emergency_vehicles', 0)))
        self.label_total_violations.setText(f"Total Violations: {stats.get('total_violations', 0)}")
        self.label_actual_violations.setText(f"Actual Violations: {stats.get('actual_violations', 0)}")
        self.label_exempted.setText(f"Exempted (Special Cases): {stats.get('exempted_count', 0)}")

        # Update collision risk stats
        risk_stats = stats.get('collision_risks', {})
        if risk_stats:
            total_risks = risk_stats.get('total_risks', 0)
            critical = risk_stats.get('critical', 0)
            high = risk_stats.get('high', 0)
            self.label_risk_count.setText(f"Active Risks: {total_risks} (Critical: {critical}, High: {high})")

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
        if plate in (None, '', '-', '识别中'):
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

            status_item = QTableWidgetItem("Review" if record.is_anomaly else "Violation")
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
            QMessageBox.critical(self, "数据库错误", str(exc))
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
            QMessageBox.information(self, "删除车辆记录", "当前仅在 vehicles 表中支持删除操作。")
            return
        rows = self.db_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "删除车辆记录", "请先在表格中选择要删除的记录。")
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
        box.setWindowTitle("确认删除")
        box.setText(f"确定要删除选中的 {len(ids)} 条车辆记录吗？此操作不可恢复。")
        box.setIcon(QMessageBox.Warning)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.setStyleSheet("""
            QMessageBox {
                background-color: #020617;
            }
            QLabel {
                color: #e5e7eb;
            }
            QPushButton {
                background-color: rgba(248, 113, 113, 0.18);
                color: #fee2e2;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: rgba(248, 113, 113, 0.26);
            }
        """)
        reply = box.exec_()
        if reply != QMessageBox.Yes:
            return

        try:
            deleted = self.database.delete_vehicles_by_ids(ids)
        except Exception as exc:
            QMessageBox.critical(self, "数据库错误", str(exc))
            return

        # 美化后的“删除完成”提示框
        info_box = QMessageBox(self)
        info_box.setWindowTitle("删除完成")
        info_box.setText(f"已删除 {deleted} 条车辆记录。")
        info_box.setIcon(QMessageBox.Information)
        info_box.setStandardButtons(QMessageBox.Ok)
        info_box.setDefaultButton(QMessageBox.Ok)
        info_box.setStyleSheet("""
            QMessageBox {
                background-color: #020617;
            }
            QLabel {
                color: #e5e7eb;
                font-size: 12px;
            }
            QPushButton {
                background-color: rgba(34, 197, 94, 0.22);
                color: #bbf7d0;
                border-radius: 6px;
                padding: 6px 16px;
                min-width: 64px;
            }
            QPushButton:hover {
                background-color: rgba(34, 197, 94, 0.30);
            }
        """)
        info_box.exec_()
        self._refresh_db_table()

    def _clean_old_db_records(self):
        """按天清理旧记录（车辆 + 违规 + 流量）"""
        days = self.db_clean_days_spin.value()
        box = QMessageBox(self)
        box.setWindowTitle("确认清理")
        box.setText(f"确定要清理 {days} 天之前的车辆、违规和流量记录吗？此操作不可恢复。")
        box.setIcon(QMessageBox.Question)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.setStyleSheet("""
            QMessageBox {
                background-color: #020617;
            }
            QLabel {
                color: #e5e7eb;
            }
            QPushButton {
                background-color: rgba(148, 163, 184, 0.15);
                color: #e5e7eb;
                border-radius: 6px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background-color: rgba(148, 163, 184, 0.22);
            }
        """)
        reply = box.exec_()
        if reply != QMessageBox.Yes:
            return
        try:
            self.database.clear_old_records(days=days)
        except Exception as exc:
            QMessageBox.critical(self, "数据库错误", str(exc))
            return
        QMessageBox.information(self, "清理完成", f"已清理早于 {days} 天的历史记录。")
        self._refresh_db_table()

    def _search_plate(self):
        """Search plate"""
        plate = self.search_input.text().strip()
        if not plate:
            return

        results = self.database.search_by_plate(plate)
        if results:
            QMessageBox.information(
                self, "Search Results",
                f"Found {len(results)} records"
            )
        else:
            QMessageBox.information(self, "Search Results", "No matching records found")

    def _reset_statistics(self):
        """Reset statistics charts"""
        self.stats_canvas.reset()
        self._frame_counter = 0
        self.statusBar.showMessage("Statistics reset")

    def _on_error(self, error: str):
        """Handle error"""
        self._is_processing = False
        self._sync_controls_for_state()
        QMessageBox.critical(self, "Error", error)
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
