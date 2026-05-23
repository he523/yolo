#!/usr/bin/env python3
"""
碰撞风险预测演示脚本
展示碰撞风险预测功能的工作原理
"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.collision_risk import CollisionRiskPredictor, RiskLevel


def demo_collision_prediction():
    """演示碰撞风险预测"""
    print("=" * 60)
    print("碰撞风险预测演示")
    print("=" * 60)

    # 创建预测器（降低阈值使其更敏感）
    predictor = CollisionRiskPredictor(
        fps=30.0,
        collision_threshold=100.0,  # 增大碰撞阈值
        ttc_thresholds={
            'critical': 1.0,
            'high': 2.0,
            'medium': 3.0,
            'low': 5.0
        }
    )

    # 创建画布
    width, height = 800, 600
    output_dir = Path('data/collision_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n场景1: 两车相向行驶（即将碰撞）")
    print("-" * 40)

    # 模拟两车相向行驶
    for frame in range(60):
        # 车辆1从左向右
        x1 = 100 + frame * 8
        # 车辆2从右向左
        x2 = 700 - frame * 8

        tracks = [
            {'track_id': 1, 'bbox': (x1, 280, x1+60, 340)},
            {'track_id': 2, 'bbox': (x2, 280, x2+60, 340)},
        ]

        risks = predictor.update(tracks)

        if frame % 10 == 0 or risks:
            # 创建可视化帧
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 40)

            # 画道路
            cv2.rectangle(canvas, (50, 250), (750, 370), (60, 60, 60), -1)
            cv2.line(canvas, (50, 310), (750, 310), (255, 255, 255), 2, cv2.LINE_AA)

            # 画车辆
            cv2.rectangle(canvas, (x1, 280), (x1+60, 340), (0, 255, 0), -1)
            cv2.putText(canvas, "V1", (x1+15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.rectangle(canvas, (x2, 280), (x2+60, 340), (0, 255, 255), -1)
            cv2.putText(canvas, "V2", (x2+15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # 画碰撞风险
            if risks:
                canvas = predictor.draw_predictions(canvas, risks)
                risk = risks[0]
                print(f"  帧 {frame}: 风险={risk.risk_level.value.upper()}, TTC={risk.time_to_collision:.2f}s")

            # 显示信息
            cv2.putText(canvas, f"Frame: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if risks:
                cv2.putText(canvas, f"RISK: {risks[0].risk_level.value.upper()}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(str(output_dir / f"scene1_frame_{frame:03d}.jpg"), canvas)

    print("\n场景2: 交叉路口（垂直交叉）")
    print("-" * 40)

    # 重置预测器
    predictor2 = CollisionRiskPredictor(
        fps=30.0,
        collision_threshold=100.0,
        ttc_thresholds={'critical': 1.0, 'high': 2.0, 'medium': 3.0, 'low': 5.0}
    )

    for frame in range(60):
        # 车辆1从左向右
        x1 = 100 + frame * 6
        # 车辆2从上向下
        y2 = 100 + frame * 6

        tracks = [
            {'track_id': 1, 'bbox': (x1, 280, x1+60, 340)},
            {'track_id': 2, 'bbox': (370, y2, 430, y2+60)},
        ]

        risks = predictor2.update(tracks)

        if frame % 10 == 0 or risks:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 40)

            # 画十字路口
            cv2.rectangle(canvas, (50, 250), (750, 370), (60, 60, 60), -1)  # 横向道路
            cv2.rectangle(canvas, (340, 50), (460, 550), (60, 60, 60), -1)  # 纵向道路

            # 画车辆
            cv2.rectangle(canvas, (x1, 280), (x1+60, 340), (0, 255, 0), -1)
            cv2.putText(canvas, "V1", (x1+15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.rectangle(canvas, (370, y2), (430, y2+60), (0, 255, 255), -1)
            cv2.putText(canvas, "V2", (385, y2+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if risks:
                canvas = predictor2.draw_predictions(canvas, risks)
                risk = risks[0]
                print(f"  帧 {frame}: 风险={risk.risk_level.value.upper()}, TTC={risk.time_to_collision:.2f}s")

            cv2.putText(canvas, f"Frame: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if risks:
                cv2.putText(canvas, f"RISK: {risks[0].risk_level.value.upper()}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(str(output_dir / f"scene2_frame_{frame:03d}.jpg"), canvas)

    print("\n场景3: 追尾风险（同向行驶，前车减速）")
    print("-" * 40)

    predictor3 = CollisionRiskPredictor(
        fps=30.0,
        collision_threshold=80.0,
        ttc_thresholds={'critical': 1.0, 'high': 2.0, 'medium': 3.0, 'low': 5.0}
    )

    for frame in range(60):
        # 后车快速行驶
        x1 = 100 + frame * 10
        # 前车慢速行驶
        x2 = 300 + frame * 3

        tracks = [
            {'track_id': 1, 'bbox': (x1, 280, x1+60, 340)},
            {'track_id': 2, 'bbox': (x2, 280, x2+60, 340)},
        ]

        risks = predictor3.update(tracks)

        if frame % 10 == 0 or risks:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (40, 40, 40)

            cv2.rectangle(canvas, (50, 250), (750, 370), (60, 60, 60), -1)
            cv2.line(canvas, (50, 310), (750, 310), (255, 255, 255), 2, cv2.LINE_AA)

            cv2.rectangle(canvas, (x1, 280), (x1+60, 340), (0, 255, 0), -1)
            cv2.putText(canvas, "V1", (x1+15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.rectangle(canvas, (x2, 280), (x2+60, 340), (0, 255, 255), -1)
            cv2.putText(canvas, "V2", (x2+15, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            if risks:
                canvas = predictor3.draw_predictions(canvas, risks)
                risk = risks[0]
                print(f"  帧 {frame}: 风险={risk.risk_level.value.upper()}, TTC={risk.time_to_collision:.2f}s")

            cv2.putText(canvas, f"Frame: {frame}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if risks:
                cv2.putText(canvas, f"RISK: {risks[0].risk_level.value.upper()}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imwrite(str(output_dir / f"scene3_frame_{frame:03d}.jpg"), canvas)

    print(f"\n演示图片保存到: {output_dir}")
    print("  - scene1_*.jpg: 相向行驶碰撞")
    print("  - scene2_*.jpg: 交叉路口碰撞")
    print("  - scene3_*.jpg: 追尾风险")

    # 生成视频
    print("\n生成演示视频...")
    for scene in [1, 2, 3]:
        images = sorted(output_dir.glob(f"scene{scene}_frame_*.jpg"))
        if images:
            frame = cv2.imread(str(images[0]))
            h, w = frame.shape[:2]
            video_path = output_dir / f"scene{scene}_demo.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 10, (w, h))
            for img_path in images:
                out.write(cv2.imread(str(img_path)))
            out.release()
            print(f"  生成: {video_path}")

    print("\n演示完成!")


if __name__ == '__main__':
    demo_collision_prediction()
