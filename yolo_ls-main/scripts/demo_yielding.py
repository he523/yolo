#!/usr/bin/env python3
"""
避让特种车辆演示脚本
模拟车辆避让救护车的场景，测试自适应违规检测的免责机制
"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.adaptive_violation import (
    AdaptiveViolationDetector, ViolationType, ExemptionReason
)
from src.core.emergency_vehicle import EmergencyVehicle, EmergencyVehicleType


def demo_yielding_exemption():
    """演示避让特种车辆免责机制"""
    print("=" * 60)
    print("避让特种车辆免责机制演示")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path('data/yielding_demo')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建检测器
    detector = AdaptiveViolationDetector(
        speed_limit=60,
        snapshot_dir=str(output_dir / 'snapshots'),
        emergency_distance=400
    )
    detector.set_stop_line(400, 100, 700)

    # 画布设置
    width, height = 800, 600
    frames = []

    print("\n场景1: 普通超速违规（无特种车辆）")
    print("-" * 40)

    # 场景1: 普通超速
    for frame_idx in range(30):
        canvas = create_road_canvas(width, height)
        x = 100 + frame_idx * 15  # 快速移动

        # 画车辆
        draw_vehicle(canvas, x, 320, "V1", (0, 255, 0))

        # 显示信息
        cv2.putText(canvas, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Scene 1: Normal Speeding", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(canvas.copy())

        if frame_idx == 20:
            # 检测违规
            record = detector.check_violation(
                track_id=1,
                bbox=(x, 300, x+60, 360),
                speed=80.0,
                frame=canvas
            )
            if record:
                print(f"  违规类型: {record.violation_type.value}")
                print(f"  速度: {record.speed} km/h")
                print(f"  是否免责: {record.is_exempted}")
                if record.is_exempted:
                    print(f"  免责原因: {record.exemption_reason.value}")

    print("\n场景2: 附近有救护车但未避让（不免责）")
    print("-" * 40)

    # 场景2: 有救护车但未避让
    detector2 = AdaptiveViolationDetector(
        speed_limit=60,
        snapshot_dir=str(output_dir / 'snapshots'),
        emergency_distance=400
    )

    for frame_idx in range(30):
        canvas = create_road_canvas(width, height)

        # 普通车辆持续快速行驶（未减速）
        x1 = 100 + frame_idx * 15

        # 救护车
        x_amb = 600 - frame_idx * 5

        draw_vehicle(canvas, x1, 320, "V1", (0, 255, 0))
        draw_ambulance(canvas, x_amb, 250)

        cv2.putText(canvas, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Scene 2: Not Yielding (No Exemption)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(canvas.copy())

        # 模拟救护车检测
        detector2.current_emergency_vehicles = [
            EmergencyVehicle(
                vehicle_type=EmergencyVehicleType.AMBULANCE,
                bbox=(x_amb, 230, x_amb+80, 290),
                confidence=0.9,
                has_warning_light=True,
                has_siren_color=True
            )
        ]

        if frame_idx == 20:
            record = detector2.check_violation(
                track_id=2,
                bbox=(x1, 300, x1+60, 360),
                speed=80.0,
                frame=canvas
            )
            if record:
                print(f"  违规类型: {record.violation_type.value}")
                print(f"  速度: {record.speed} km/h")
                print(f"  是否免责: {record.is_exempted}")
                print(f"  附近特种车辆: {record.nearby_emergency_vehicles}")
                if not record.is_exempted:
                    print("  原因: 虽然附近有救护车，但车辆未表现出避让行为")

    print("\n场景3: 避让救护车（减速）- 免责")
    print("-" * 40)

    # 场景3: 避让救护车 - 减速
    detector3 = AdaptiveViolationDetector(
        speed_limit=60,
        snapshot_dir=str(output_dir / 'snapshots'),
        emergency_distance=400
    )

    positions = []
    for frame_idx in range(40):
        canvas = create_road_canvas(width, height)

        # 车辆先快后慢（减速避让）- 更明显的速度变化
        if frame_idx < 15:
            speed = 25  # 快速
        else:
            speed = 3   # 大幅减速

        if frame_idx == 0:
            x1 = 100
        else:
            x1 = positions[-1] + speed

        positions.append(x1)

        # 救护车
        x_amb = 700 - frame_idx * 8

        draw_vehicle(canvas, x1, 320, "V1", (0, 255, 0))
        draw_ambulance(canvas, x_amb, 250)

        # 显示速度变化
        cv2.putText(canvas, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Scene 3: Yielding (Deceleration)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, f"Speed: {speed} px/frame", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        frames.append(canvas.copy())

        # 模拟救护车检测
        detector3.current_emergency_vehicles = [
            EmergencyVehicle(
                vehicle_type=EmergencyVehicleType.AMBULANCE,
                bbox=(x_amb, 230, x_amb+80, 290),
                confidence=0.9,
                has_warning_light=True,
                has_siren_color=True
            )
        ]

        # 更新轨迹历史 - 使用实际位置
        center = (x1 + 30, 330)
        if 3 not in detector3.track_history:
            from collections import deque
            detector3.track_history[3] = deque(maxlen=20)  # 增加历史长度
            detector3.recorded_violations[3] = set()
        detector3.track_history[3].append(center)

        if frame_idx == 20:  # 在刚开始减速时检测（历史中包含快速和慢速数据）
            record = detector3.check_violation(
                track_id=3,
                bbox=(x1, 300, x1+60, 360),
                speed=75.0,
                frame=canvas
            )
            if record:
                print(f"  违规类型: {record.violation_type.value}")
                print(f"  速度: {record.speed} km/h")
                print(f"  是否免责: {record.is_exempted}")
                print(f"  附近特种车辆: {record.nearby_emergency_vehicles}")
                if record.is_exempted:
                    print(f"  免责原因: {record.exemption_reason.value}")
                    print(f"  详情: {record.exemption_details}")

    print("\n场景4: 避让救护车（变道）- 免责")
    print("-" * 40)

    # 场景4: 避让救护车 - 变道
    detector4 = AdaptiveViolationDetector(
        speed_limit=60,
        snapshot_dir=str(output_dir / 'snapshots'),
        emergency_distance=400
    )

    for frame_idx in range(40):
        canvas = create_road_canvas(width, height)

        # 车辆变道避让
        x1 = 100 + frame_idx * 10
        if frame_idx < 15:
            y1 = 320
        else:
            y1 = 320 + (frame_idx - 15) * 8  # 向下变道

        # 救护车
        x_amb = 700 - frame_idx * 12

        draw_vehicle(canvas, x1, y1, "V1", (0, 255, 0))
        draw_ambulance(canvas, x_amb, 320)

        cv2.putText(canvas, f"Frame: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Scene 4: Yielding (Lane Change)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frames.append(canvas.copy())

        detector4.current_emergency_vehicles = [
            EmergencyVehicle(
                vehicle_type=EmergencyVehicleType.AMBULANCE,
                bbox=(x_amb, 300, x_amb+80, 360),
                confidence=0.9,
                has_warning_light=True,
                has_siren_color=True
            )
        ]

        center = (x1 + 30, y1 + 20)
        if 4 not in detector4.track_history:
            from collections import deque
            detector4.track_history[4] = deque(maxlen=10)
            detector4.recorded_violations[4] = set()
        detector4.track_history[4].append(center)

        if frame_idx == 30:
            record = detector4.check_violation(
                track_id=4,
                bbox=(x1, y1, x1+60, y1+40),
                speed=70.0,
                frame=canvas
            )
            if record:
                print(f"  违规类型: {record.violation_type.value}")
                print(f"  速度: {record.speed} km/h")
                print(f"  是否免责: {record.is_exempted}")
                print(f"  附近特种车辆: {record.nearby_emergency_vehicles}")
                if record.is_exempted:
                    print(f"  免责原因: {record.exemption_reason.value}")
                    print(f"  详情: {record.exemption_details}")

    # 保存视频
    print(f"\n保存演示视频到: {output_dir}")
    video_path = output_dir / "yielding_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 15, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"  生成: {video_path}")

    # 统计
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("场景1: 普通超速 -> 违规（无免责）")
    print("场景2: 附近有救护车但未避让 -> 违规（无免责）")
    print("场景3: 减速避让救护车 -> 免责")
    print("场景4: 变道避让救护车 -> 免责")
    print("\n关键点: 仅仅附近有特种车辆不足以免责，必须有实际的避让行为！")


def create_road_canvas(width, height):
    """创建道路背景"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)

    # 画道路
    cv2.rectangle(canvas, (50, 200), (750, 450), (60, 60, 60), -1)

    # 车道线
    cv2.line(canvas, (50, 280), (750, 280), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (50, 370), (750, 370), (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def draw_vehicle(canvas, x, y, label, color):
    """画普通车辆"""
    cv2.rectangle(canvas, (x, y), (x+60, y+40), color, -1)
    cv2.putText(canvas, label, (x+15, y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_ambulance(canvas, x, y):
    """画救护车"""
    # 车身（白色）
    cv2.rectangle(canvas, (x, y), (x+80, y+50), (255, 255, 255), -1)
    # 红十字
    cv2.rectangle(canvas, (x+35, y+10), (x+45, y+40), (0, 0, 255), -1)
    cv2.rectangle(canvas, (x+25, y+20), (x+55, y+30), (0, 0, 255), -1)
    # 警示灯
    cv2.circle(canvas, (x+20, y-5), 8, (0, 0, 255), -1)
    cv2.circle(canvas, (x+60, y-5), 8, (255, 0, 0), -1)
    # 标签
    cv2.putText(canvas, "AMB", (x+20, y+65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


if __name__ == '__main__':
    demo_yielding_exemption()
