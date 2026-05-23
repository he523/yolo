#!/usr/bin/env python3
"""
批量检测脚本 - 对视频进行车辆检测、跟踪和碰撞风险预测
用于精准度测试
"""
import sys
import cv2
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import VehicleDetector, ByteTracker
from src.core.collision_risk import CollisionRiskPredictor, RiskLevel


def process_video(video_path: str, output_dir: str, sample_interval: int = 30):
    """
    处理视频并保存检测结果（包含碰撞风险预测）
    """
    video_name = Path(video_path).stem
    result_dir = Path(output_dir) / video_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # 初始化检测器、跟踪器和碰撞风险预测器
    detector = VehicleDetector(
        model_path='models/yolo12n.pt',
        confidence=0.5,
        device='cpu'
    )
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化碰撞风险预测器
    collision_predictor = CollisionRiskPredictor(fps=fps)

    print(f"\n处理视频: {video_name}")
    print(f"  FPS: {fps}, 总帧数: {total_frames}")
    print(f"  抽帧间隔: {sample_interval} (约每 {sample_interval/fps:.1f} 秒)")

    frame_idx = 0
    saved_count = 0
    all_results = []
    all_risks = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测和跟踪
        detections = detector.detect_vehicles(frame)
        tracks = tracker.update(detections)

        # 准备轨迹数据用于碰撞预测
        track_data = [
            {'track_id': t.track_id, 'bbox': t.bbox}
            for t in tracks
        ]

        # 碰撞风险预测（每帧都更新以积累轨迹历史）
        collision_risks = collision_predictor.update(track_data)

        # 按间隔保存
        if frame_idx % sample_interval == 0:
            # 保存原始帧
            orig_path = result_dir / f"frame_{frame_idx:05d}_orig.jpg"
            cv2.imwrite(str(orig_path), frame)

            # 收集有风险的车辆ID及其风险等级
            risk_vehicles = {}  # {track_id: risk_level}
            for risk in collision_risks:
                for vid in [risk.vehicle1_id, risk.vehicle2_id]:
                    if vid not in risk_vehicles or risk.risk_level.value < risk_vehicles[vid].value:
                        risk_vehicles[vid] = risk.risk_level

            # 风险等级对应的颜色 (BGR)
            risk_colors = {
                RiskLevel.SAFE: (0, 255, 0),      # 绿色
                RiskLevel.LOW: (0, 255, 255),     # 黄色
                RiskLevel.MEDIUM: (0, 165, 255),  # 橙色
                RiskLevel.HIGH: (0, 0, 255),      # 红色
                RiskLevel.CRITICAL: (255, 0, 255) # 紫色
            }

            # 绘制检测结果
            annotated = frame.copy()
            frame_detections = []

            for track in tracks:
                x1, y1, x2, y2 = track.bbox

                # 根据风险等级选择颜色
                if track.track_id in risk_vehicles:
                    color = risk_colors[risk_vehicles[track.track_id]]
                    thickness = 3  # 有风险的框加粗
                else:
                    color = (0, 255, 0)  # 安全-绿色
                    thickness = 2

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                # 标签也用相同颜色
                label = f"ID:{track.track_id}"
                if track.track_id in risk_vehicles:
                    label += f" [{risk_vehicles[track.track_id].value.upper()}]"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                frame_detections.append({
                    'track_id': track.track_id,
                    'class': track.class_name,
                    'confidence': float(track.confidence),
                    'bbox': [x1, y1, x2, y2],
                    'risk': risk_vehicles.get(track.track_id, RiskLevel.SAFE).value
                })

            # 绘制碰撞风险连线和预测轨迹
            if collision_risks:
                annotated = collision_predictor.draw_predictions(annotated, collision_risks)

            # 保存标注帧
            annot_path = result_dir / f"frame_{frame_idx:05d}_detect.jpg"
            cv2.imwrite(str(annot_path), annotated)

            # 记录碰撞风险
            frame_risks = []
            for risk in collision_risks:
                frame_risks.append({
                    'vehicle1_id': risk.vehicle1_id,
                    'vehicle2_id': risk.vehicle2_id,
                    'risk_level': risk.risk_level.value,
                    'ttc': risk.time_to_collision,
                    'confidence': risk.confidence
                })
                all_risks.append({
                    'frame': frame_idx,
                    'risk': frame_risks[-1]
                })

            # 记录结果
            all_results.append({
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'detections': frame_detections,
                'detection_count': len(frame_detections),
                'collision_risks': frame_risks
            })

            saved_count += 1
            risk_str = f", 风险: {len(collision_risks)}" if collision_risks else ""
            print(f"  帧 {frame_idx}: 检测 {len(tracks)} 个目标{risk_str}")

        frame_idx += 1

    cap.release()

    # 统计碰撞风险
    risk_stats = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    for r in all_risks:
        level = r['risk']['risk_level']
        if level in risk_stats:
            risk_stats[level] += 1

    # 保存JSON结果
    summary = {
        'video': video_name,
        'total_frames': total_frames,
        'fps': fps,
        'sample_interval': sample_interval,
        'saved_frames': saved_count,
        'results': all_results,
        'statistics': {
            'total_detections': sum(r['detection_count'] for r in all_results),
            'avg_detections_per_frame': sum(r['detection_count'] for r in all_results) / max(saved_count, 1),
            'max_detections': max((r['detection_count'] for r in all_results), default=0)
        },
        'collision_risk_statistics': {
            'total_risks': len(all_risks),
            **risk_stats
        }
    }

    json_path = result_dir / 'results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  结果保存到: {result_dir}")
    print(f"  总检测数: {summary['statistics']['total_detections']}")
    print(f"  平均每帧: {summary['statistics']['avg_detections_per_frame']:.1f}")
    print(f"  碰撞风险: {len(all_risks)} (Critical:{risk_stats['critical']}, High:{risk_stats['high']}, Medium:{risk_stats['medium']}, Low:{risk_stats['low']})")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='批量车辆检测与碰撞风险预测')
    parser.add_argument('--input', '-i', required=True, help='视频文件或目录')
    parser.add_argument('--output', '-o', default='data/detection_results', help='输出目录')
    parser.add_argument('--interval', '-n', type=int, default=30, help='抽帧间隔')
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        videos = [input_path]
    elif input_path.is_dir():
        videos = list(input_path.glob('*.mp4')) + list(input_path.glob('*.avi'))
    else:
        print(f"无效路径: {input_path}")
        return

    print(f"找到 {len(videos)} 个视频文件")
    print("功能: 车辆检测 + 跟踪 + 碰撞风险预测")

    for video in videos:
        process_video(str(video), args.output, args.interval)

    print("\n检测完成!")
    print(f"结果保存在: {args.output}")
    print("\n你可以查看:")
    print("  - *_orig.jpg: 原始帧")
    print("  - *_detect.jpg: 检测结果帧（含碰撞风险可视化）")
    print("  - results.json: 详细检测和风险数据")


if __name__ == '__main__':
    main()
