#!/usr/bin/env python3
"""
实时交通分析系统 - 主程序入口

基于 YOLOv12 + ByteTrack 的智能交通监控系统
功能：车辆检测、跟踪、特征提取、违规识别、数据可视化
"""
import json
import os
import sys
from collections import Counter

# 在导入任何Qt相关模块之前设置环境变量，避免OpenCV Qt插件冲突
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''  # 清除OpenCV的Qt路径
os.environ.pop('QT_PLUGIN_PATH', None)  # 移除可能的冲突路径

# Windows：禁用 Paddle PIR/oneDNN，避免 OCR 推理崩溃
if sys.platform == 'win32':
    os.environ['FLAGS_enable_pir_api'] = '0'
    os.environ['FLAGS_enable_pir_in_executor'] = '0'
    os.environ['FLAGS_use_mkldnn'] = '0'
    os.environ['FLAGS_json_format_model'] = '0'
    os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'

import argparse
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def run_gui():
    """运行 GUI 模式"""
    from PyQt5.QtWidgets import QApplication
    from src.gui import MainWindow

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


def _serialize_risks(collision_risks):
    """序列化碰撞风险为JSON可写结构"""
    serialized = []
    for risk in collision_risks:
        serialized.append({
            'vehicle1_id': int(risk.vehicle1_id),
            'vehicle2_id': int(risk.vehicle2_id),
            'risk_level': risk.risk_level.value,
            'time_to_collision': float(risk.time_to_collision),
            'confidence': float(risk.confidence),
            'collision_point': (
                [float(risk.collision_point[0]), float(risk.collision_point[1])]
                if risk.collision_point
                else None
            ),
        })
    return serialized


def run_cli(args):
    """运行命令行模式（使用自适应违规检测）"""
    import cv2
    import time
    from src.video import VideoStream
    from src.core import (
        VehicleDetector,
        ByteTracker,
        FeatureExtractor,
        AdaptiveViolationDetector,
        VehicleInteractionGraph,
        CollisionRiskPredictor,
        RiskLevel,
    )
    from src.database import Database
    from src.database.scheduler import start_db_cleanup_from_config
    from src.ocr import PlateReader
    from src.ocr.ocr_scheduler import OCRScheduler
    from src.utils import load_config, setup_logging, ModelManager
    from src.utils.performance import PerformanceOptimizer, FPSMonitor

    config = load_config(args.config)
    sys_cfg = config.get('system', {})
    setup_logging(sys_cfg.get('log_level', 'INFO'))
    risk_cfg = config.get('risk', {})
    vio_cfg = config.get('violation', {})

    if args.enable_risk:
        risk_enabled = True
    elif args.disable_risk:
        risk_enabled = False
    else:
        risk_enabled = risk_cfg.get('enabled', True)

    collision_model_path = args.collision_model or risk_cfg.get('collision_model_path')
    stgat_model_path = args.stgat_model or risk_cfg.get('stgat_model_path')

    video_cfg = config.get('video', {})
    det_cfg = config.get('detector', {})
    tracker_cfg = config.get('tracker', {})
    ocr_cfg = config.get('ocr', {})

    video_source = args.source
    if video_source == '0' and video_cfg.get('source') not in (None, ''):
        video_source = str(video_cfg.get('source'))

    confidence = det_cfg.get('confidence', args.confidence)
    base_imgsz = int(det_cfg.get('imgsz', 768))

    # 初始化组件
    video = VideoStream(
        video_source,
        fps=video_cfg.get('fps', 15),
    )
    model_mgr = ModelManager(config)
    model_path, _ = model_mgr.load_yolo_path(args.model or det_cfg.get('model_path'))
    perf_cfg = config.get('performance', {})
    perf_enabled = perf_cfg.get('enabled', True)
    perf = PerformanceOptimizer.from_config(
        perf_cfg,
        base_imgsz=base_imgsz,
    )
    perf._plan.enable_tiling = det_cfg.get('enable_tiling', False)
    perf.base_enable_tiling = det_cfg.get('enable_tiling', False)
    target_fps = perf_cfg.get('target_fps', config.get('video', {}).get('fps', 15))
    fps_monitor = (
        FPSMonitor(
            target_fps=target_fps,
            warmup_frames=int(perf_cfg.get('warmup_frames', 30)),
            low_fps_checks=int(perf_cfg.get('low_fps_checks', 5)),
        )
        if perf_enabled else None
    )
    detector = VehicleDetector(
        model_path=model_path,
        confidence=confidence,
        iou_threshold=det_cfg.get('iou_threshold', 0.45),
        device=args.device,
        imgsz=base_imgsz,
        max_det=det_cfg.get('max_det', 300),
        enable_tiling=det_cfg.get('enable_tiling', False),
        tiling_grid=tuple(det_cfg.get('tiling_grid', [2, 2])),
        tiling_overlap=det_cfg.get('tiling_overlap', 0.20),
        tiling_min_dets=det_cfg.get('tiling_min_dets', 10),
        tiling_interval=det_cfg.get('tiling_interval', 5),
        tiling_mode=det_cfg.get('tiling_mode', 'strip'),
    )
    tracker = ByteTracker(
        track_thresh=tracker_cfg.get('track_thresh', 0.5),
        track_buffer=tracker_cfg.get('track_buffer', 30),
        match_thresh=tracker_cfg.get('match_thresh', 0.8),
        min_box_area=tracker_cfg.get('min_box_area', 10),
    )
    feature_extractor = FeatureExtractor(
        pixel_to_meter=config.get('feature', {}).get('pixel_to_meter', 0.05),
        fps=config.get('video', {}).get('fps', 15)
    )
    violation_detector = AdaptiveViolationDetector(
        speed_limit=vio_cfg.get('speed_limit', 60),
        snapshot_dir=vio_cfg.get('snapshot_dir', 'data/snapshots'),
        emergency_distance=vio_cfg.get('emergency_distance', 300),
        wrong_way_enabled=vio_cfg.get('wrong_way_enabled', True),
        illegal_lane_enabled=vio_cfg.get('illegal_lane_enabled', True),
        expected_flow_direction=vio_cfg.get('expected_flow_direction', 'south'),
        lane_change_lateral_px=vio_cfg.get('lane_change_lateral_px', 80),
        lane_change_min_speed_kmh=vio_cfg.get('lane_change_min_speed_kmh', 15),
    )
    # 如果在配置中预先定义了停止线，则启用闯红灯检测
    stop_line_cfg = vio_cfg.get('stop_line')
    if stop_line_cfg:
        try:
            violation_detector.set_stop_line(
                y=int(stop_line_cfg.get('y', 0)),
                x_start=int(stop_line_cfg.get('x_start', 0)),
                x_end=int(stop_line_cfg.get('x_end', 0)),
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Invalid stop_line in config: %s", exc)

    plate_reader = None
    if ocr_cfg.get('enabled', True) and int(ocr_cfg.get('interval', 15)) > 0:
        plate_reader = PlateReader(
            model_path=ocr_cfg.get('model_path', 'models/plate_ocr.pt'),
            use_gpu=ocr_cfg.get('use_gpu', args.device != 'cpu'),
            paddle_mobile=ocr_cfg.get('paddle_mobile', True),
        )
    db_cfg = config.get('database', {})
    database = Database(
        db_path=db_cfg.get('path', 'data/traffic.db'),
        pool_size=int(db_cfg.get('pool_size', 5)),
    )
    db_cleanup_scheduler = start_db_cleanup_from_config(database, config)

    interaction_graph = None
    collision_predictor = None
    if risk_enabled:
        interaction_graph = VehicleInteractionGraph(
            distance_threshold=risk_cfg.get('interaction_distance', 200),
            temporal_window=risk_cfg.get('temporal_window', 10),
            model_path=stgat_model_path,
            device=args.device,
        )
        collision_predictor = CollisionRiskPredictor(
            history_length=risk_cfg.get('history_length', 10),
            prediction_horizon=risk_cfg.get('prediction_horizon', 15),
            fps=config.get('video', {}).get('fps', 15),
            collision_threshold=risk_cfg.get('collision_threshold', 150.0),
            ttc_thresholds=risk_cfg.get('ttc_thresholds'),
            model_path=collision_model_path,
            device=args.device,
        )

    if not video.open():
        print(f"Error: Cannot open video source: {args.source}")
        return 1

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = video.get_frame_size()
        fps = video.get_fps()
        if fps <= 0:
            fps = config.get('video', {}).get('fps', 15)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"Warning: Cannot open output writer: {output_path}")
            writer = None

    result_writer = None
    if args.risk_output:
        result_path = Path(args.risk_output)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_writer = result_path.open('w', encoding='utf-8')

    print(f"Processing video: {video_source}")
    print(f"Model: {args.model}, Device: {args.device}")
    if writer:
        print(f"Output video: {args.output}")
    if result_writer:
        print(f"Output structured results: {args.risk_output}")
    print("自适应违规检测已启用（支持特种车辆避让免责）")
    print(f"碰撞风险检测: {'启用' if risk_enabled else '关闭'}")
    print("Press 'q' to quit")

    frame_count = 0
    plate_cache = {}
    ocr_interval = int(ocr_cfg.get('interval', 15))
    ocr_min_h = int(ocr_cfg.get('min_bbox_height', 40))
    ocr_max_per_frame = int(ocr_cfg.get('max_vehicles_per_frame', 8))
    ocr_scheduler = OCRScheduler(max_per_frame=ocr_max_per_frame)
    seen_tracks = set()
    risk_level_counter = Counter()
    max_active_risks = 0
    video_fps = video.get_fps()
    if video_fps <= 0:
        video_fps = config.get('video', {}).get('fps', 15)

    base_risk_interval = int(risk_cfg.get('interval', 3))

    while True:
        frame_start = time.perf_counter()
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        if perf_enabled and not perf.should_process_frame(frame_count):
            continue

        if perf_enabled:
            detector.imgsz = perf.get_imgsz()
            detector.enable_tiling = perf.get_enable_tiling()

        effective_risk_interval = (
            perf.get_risk_interval(base_risk_interval) if perf_enabled else base_risk_interval
        )
        effective_ocr_interval = (
            perf.get_ocr_interval(ocr_interval) if perf_enabled else ocr_interval
        )
        risk_active = risk_enabled and not (perf_enabled and perf.risk_disabled())

        # 检测与跟踪（性能优化：尽量单次推理，避免每帧多次 predict）
        det_cfg = config.get('detector', {})
        enable_context = bool(det_cfg.get('enable_context_detection', False))
        if enable_context:
            all_classes = list(detector.VEHICLE_CLASSES.keys()) + [
                detector.PERSON_CLASS,
                detector.TRAFFIC_LIGHT_CLASS,
            ]
            all_dets = detector.detect(frame, all_classes)
            detections = [d for d in all_dets if d.class_id in detector.VEHICLE_CLASSES]
            person_bboxes = [d.bbox for d in all_dets if d.class_id == detector.PERSON_CLASS]
            light_dets = [d for d in all_dets if d.class_id == detector.TRAFFIC_LIGHT_CLASS]
            light_bbox = max(light_dets, key=lambda d: d.confidence).bbox if light_dets else None
        else:
            detections = detector.detect_vehicles(frame)
            person_bboxes = []
            light_bbox = None

        tracks = tracker.update(detections)
        track_data = [{'track_id': t.track_id, 'bbox': t.bbox} for t in tracks]

        interaction_embeddings = {}
        collision_risks = []
        risk_summary = {'total_risks': 0, 'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'min_ttc': -1}
        if risk_active and interaction_graph and collision_predictor:
            risk_max_tracks = int(risk_cfg.get('max_tracks', 20))
            if effective_risk_interval <= 0:
                effective_risk_interval = 1

            if frame_count % effective_risk_interval == 0:
                def _area(td):
                    x1, y1, x2, y2 = td['bbox']
                    return max(0, (x2 - x1) * (y2 - y1))

                track_data_risk = sorted(track_data, key=_area, reverse=True)[:max(1, risk_max_tracks)]
                interaction_embeddings = interaction_graph.update(track_data_risk)
                collision_risks = collision_predictor.update(track_data_risk)
            else:
                interaction_embeddings = {}
                collision_risks = []
            risk_summary = collision_predictor.get_risk_summary(collision_risks)
            for risk in collision_risks:
                risk_level_counter[risk.risk_level.value] += 1
            max_active_risks = max(max_active_risks, len(collision_risks))

        # 更新违规检测上下文
        vehicle_bboxes = [t.bbox for t in tracks]
        violation_detector.update(
            frame,
            vehicle_bboxes,
            person_bboxes=person_bboxes,
            light_bbox=light_bbox
        )

        frame_tracks = []
        frame_violations = []

        ocr_targets = []
        if plate_reader and effective_ocr_interval > 0 and frame_count % effective_ocr_interval == 0:
            bbox_heights = {
                t.track_id: t.bbox[3] - t.bbox[1] for t in tracks
            }
            ocr_targets = ocr_scheduler.select_tracks(
                [t.track_id for t in tracks],
                plate_cache,
                bbox_heights,
                ocr_min_h,
            )

        for track in tracks:
            plate_number = plate_cache.get(track.track_id)
            if plate_reader and track.track_id in ocr_targets:
                plate_result = plate_reader.read(frame, track.bbox)
                if plate_result:
                    plate_number = plate_result.plate_number
                    plate_cache[track.track_id] = plate_number

            features = feature_extractor.extract(frame, track.track_id, track.bbox)
            frame_tracks.append({
                'track_id': int(track.track_id),
                'class_name': track.class_name,
                'bbox': [int(v) for v in track.bbox],
                'speed_kmh': float(features.speed),
                'direction': features.direction.value,
                'color': features.color,
                'plate_number': plate_number,
            })

            # 维护车辆记录，支持车牌检索
            if track.track_id in seen_tracks:
                database.update_vehicle(
                    track_id=track.track_id,
                    speed=features.speed,
                    direction=features.direction.value,
                    plate_number=plate_number,
                    vehicle_type=track.class_name,
                    color=features.color,
                )
            else:
                database.add_vehicle(
                    track_id=track.track_id,
                    plate_number=plate_number,
                    vehicle_type=track.class_name,
                    color=features.color,
                    speed=features.speed,
                    direction=features.direction.value,
                )
                seen_tracks.add(track.track_id)

            record = violation_detector.check_violation(
                track_id=track.track_id,
                bbox=track.bbox,
                speed=features.speed,
                frame=frame,
                plate_number=plate_number,
                direction=features.direction,
            )

            if record:
                frame_violations.append({
                    'record_id': record.record_id,
                    'track_id': int(record.track_id),
                    'violation_type': record.violation_type.value,
                    'is_anomaly': bool(record.is_anomaly),
                    'anomaly_reason': record.anomaly_reason.value if record.is_anomaly else None,
                    'speed_kmh': float(record.speed) if record.speed is not None else None,
                    'location': [int(record.location[0]), int(record.location[1])],
                    'plate_number': record.plate_number if record.plate_number not in ("识别中", "-") else None,
                    'snapshot_path': record.snapshot_path,
                })
                database.add_violation(
                    track_id=record.track_id,
                    violation_type=record.violation_type.value,
                    location=record.location,
                    speed=record.speed,
                    plate_number=record.plate_number,
                    snapshot_path=record.snapshot_path,
                    record_id=record.record_id,
                    is_exempted=record.is_anomaly,
                    exemption_reason=record.anomaly_reason.value if record.is_anomaly else None,
                    exemption_details=", ".join(record.nearby_objects) if record.nearby_objects else None,
                    nearby_emergency_vehicles=record.nearby_objects,
                )

        if frame_count % 30 == 0 and frame_tracks:
            avg_speed = sum(item['speed_kmh'] for item in frame_tracks) / len(frame_tracks)
            direction_counter = Counter(item['direction'] for item in frame_tracks)
            dominant_direction = direction_counter.most_common(1)[0][0]
            database.add_traffic_flow(
                vehicle_count=len(frame_tracks),
                avg_speed=avg_speed,
                direction=dominant_direction,
            )

        # 绘制标注
        annotated = violation_detector.draw_annotations(frame)
        if risk_active and collision_predictor and collision_risks:
            annotated = collision_predictor.draw_predictions(annotated, collision_risks, track_data)

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plate = plate_cache.get(track.track_id)
            if plate:
                cv2.putText(
                    annotated,
                    plate,
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

        stats = violation_detector.get_statistics()
        cv2.putText(
            annotated,
            f"Vehicles: {len(tracks)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Violations: {stats['actual_violations']} | Exempted: {stats['exempted_count']}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if risk_active:
            cv2.putText(
                annotated,
                f"Risk C/H/M/L: {risk_summary.get('critical', 0)}/{risk_summary.get('high', 0)}/{risk_summary.get('medium', 0)}/{risk_summary.get('low', 0)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 0),
                2,
            )

        if perf_enabled and fps_monitor and frame_count % 30 == 0:
            status = perf.get_status()
            cv2.putText(
                annotated,
                f"FPS:{fps_monitor.avg_fps:.1f} imgsz:{status['imgsz']} L{status['degradation_level']}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
            )

        if writer:
            writer.write(annotated)

        if result_writer:
            payload = {
                'frame_index': frame_count,
                'timestamp_sec': round(frame_count / max(video_fps, 1e-6), 3),
                'vehicle_count': len(frame_tracks),
                'violation_count': len(frame_violations),
                'interaction_embeddings': len(interaction_embeddings),
                'tracks': frame_tracks,
                'violations': frame_violations,
                'collision_risks': _serialize_risks(collision_risks),
                'risk_summary': {
                    'total_risks': int(risk_summary.get('total_risks', 0)),
                    'critical': int(risk_summary.get('critical', 0)),
                    'high': int(risk_summary.get('high', 0)),
                    'medium': int(risk_summary.get('medium', 0)),
                    'low': int(risk_summary.get('low', 0)),
                    'min_ttc': float(risk_summary.get('min_ttc', -1)),
                    'highest_risk_pair': risk_summary.get('highest_risk_pair'),
                },
            }
            result_writer.write(json.dumps(payload, ensure_ascii=False) + '\n')

        if not args.headless:
            cv2.imshow("Traffic Analysis - Adaptive Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if perf_enabled and fps_monitor:
            elapsed = time.perf_counter() - frame_start
            fps_monitor.tick(elapsed)
            plan = fps_monitor.check_performance()
            if plan:
                perf.apply_degradation(plan)

    video.release()
    if db_cleanup_scheduler is not None:
        db_cleanup_scheduler.stop()
    database.close()
    if writer:
        writer.release()
    if result_writer:
        result_writer.close()
    cv2.destroyAllWindows()

    final_stats = violation_detector.get_statistics()
    print(f"\n=== 处理完成 ===")
    print(f"处理帧数: {frame_count}")
    print(f"总违规数: {final_stats['total_violations']}")
    print(f"实际违规: {final_stats['actual_violations']}")
    print(f"特殊情况(免责): {final_stats['exempted_count']}")
    if risk_enabled:
        print(
            "累计风险事件(C/H/M/L): "
            f"{risk_level_counter[RiskLevel.CRITICAL.value]}/"
            f"{risk_level_counter[RiskLevel.HIGH.value]}/"
            f"{risk_level_counter[RiskLevel.MEDIUM.value]}/"
            f"{risk_level_counter[RiskLevel.LOW.value]}"
        )
        print(f"峰值并发风险数: {max_active_risks}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="实时交通分析系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--gui', action='store_true',
        help='启动图形界面模式'
    )
    parser.add_argument(
        '--source', type=str, default='0',
        help='视频源（摄像头ID、RTSP地址或视频文件路径）'
    )
    parser.add_argument(
        '--model', type=str, default='models/yolo12n_vehicle.pt',
        help='YOLOv12 model path'
    )
    parser.add_argument(
        '--confidence', type=float, default=0.2,
        help='检测置信度阈值'
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cuda', 'cpu'],
        help='运行设备'
    )
    parser.add_argument(
        '--config', type=str, default='config/settings.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='输出视频路径'
    )
    parser.add_argument(
        '--risk-output', type=str, default=None,
        help='输出逐帧结构化结果(JSONL)路径'
    )
    parser.add_argument(
        '--collision-model', type=str, default=None,
        help='碰撞预测模型权重路径(.pt)'
    )
    parser.add_argument(
        '--stgat-model', type=str, default=None,
        help='ST-GAT模型权重路径(.pt)'
    )
    risk_group = parser.add_mutually_exclusive_group()
    risk_group.add_argument(
        '--enable-risk', action='store_true',
        help='强制启用碰撞风险检测'
    )
    risk_group.add_argument(
        '--disable-risk', action='store_true',
        help='强制关闭碰撞风险检测'
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='无头模式（不显示窗口）'
    )

    args = parser.parse_args()

    if args.gui:
        run_gui()
    else:
        sys.exit(run_cli(args))


if __name__ == '__main__':
    main()
