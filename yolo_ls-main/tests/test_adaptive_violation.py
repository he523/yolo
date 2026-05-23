#!/usr/bin/env python3
"""
自适应违规检测系统测试脚本
测试特种车辆检测和异常标记功能
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.emergency_vehicle import EmergencyVehicleDetector, EmergencyVehicleType, EmergencyVehicle
from src.core.adaptive_violation import (
    AdaptiveViolationDetector, ViolationType, AnomalyReason, ANOMALY_DESCRIPTIONS
)


def create_test_frame(width=640, height=480):
    """创建测试帧"""
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_emergency_vehicle_detector():
    """测试特种车辆检测"""
    print("=" * 50)
    print("Test 1: Emergency Vehicle Detector")
    print("=" * 50)

    detector = EmergencyVehicleDetector()

    # 测试空帧
    normal_frame = create_test_frame()
    bboxes = [(100, 100, 200, 200)]
    results = detector.detect(normal_frame, bboxes)
    print(f"Normal frame: {len(results)} emergency vehicles")
    assert len(results) == 0
    print("[PASS] No false positive")

    return True


def test_adaptive_violation_detector():
    """测试自适应违规检测"""
    print("\n" + "=" * 50)
    print("Test 2: Adaptive Violation Detector")
    print("=" * 50)

    detector = AdaptiveViolationDetector(
        speed_limit=60,
        snapshot_dir='data/test_snapshots',
        emergency_distance=300
    )

    frame = create_test_frame()

    # 测试1: 普通超速违规
    print("\n2.1 Testing normal speeding violation...")
    record = detector.check_violation(
        track_id=1,
        bbox=(100, 100, 200, 200),
        speed=80.0,
        frame=frame
    )

    if record:
        print(f"  Violation type: {record.violation_type.value}")
        print(f"  Speed: {record.speed}")
        print(f"  Is anomaly: {record.is_anomaly}")
        assert record.violation_type == ViolationType.SPEEDING
        assert not record.is_anomaly
        print("[PASS] Normal speeding violation detected")
    else:
        print("[FAIL] Violation not detected")
        return False

    # 测试2: 附近有特种车辆时的违规（标记为异常）
    print("\n2.2 Testing anomaly with emergency vehicle nearby...")

    detector.current_emergency_vehicles = [
        EmergencyVehicle(
            vehicle_type=EmergencyVehicleType.AMBULANCE,
            bbox=(150, 100, 250, 200),
            confidence=0.9,
            has_warning_light=True,
            has_siren_color=True
        )
    ]

    record = detector.check_violation(
        track_id=2,
        bbox=(100, 100, 200, 200),
        speed=75.0,
        frame=frame
    )

    if record:
        print(f"  Violation type: {record.violation_type.value}")
        print(f"  Is anomaly: {record.is_anomaly}")
        print(f"  Anomaly reason: {record.anomaly_reason.value}")
        print(f"  Nearby objects: {record.nearby_objects}")
        assert record.is_anomaly
        assert record.anomaly_reason == AnomalyReason.EMERGENCY_VEHICLE
        print("[PASS] Anomaly marked correctly")
    else:
        print("[FAIL] Violation not detected")
        return False

    detector.current_emergency_vehicles = []
    return True


def test_statistics():
    """测试统计功能"""
    print("\n" + "=" * 50)
    print("Test 3: Statistics Tracking")
    print("=" * 50)

    detector = AdaptiveViolationDetector(speed_limit=60)
    frame = create_test_frame()

    # 生成普通违规
    for i in range(3):
        detector.check_violation(
            track_id=10 + i,
            bbox=(100, 100, 200, 200),
            speed=70.0 + i * 5,
            frame=frame
        )

    # 添加特种车辆，生成异常
    detector.current_emergency_vehicles = [
        EmergencyVehicle(
            vehicle_type=EmergencyVehicleType.FIRE_TRUCK,
            bbox=(150, 100, 250, 200),
            confidence=0.9,
            has_warning_light=True,
            has_siren_color=True
        )
    ]

    for i in range(2):
        detector.check_violation(
            track_id=20 + i,
            bbox=(100, 100, 200, 200),
            speed=65.0 + i * 5,
            frame=frame
        )

    stats = detector.get_statistics()
    print(f"Total violations: {stats['total_violations']}")
    print(f"Anomaly count: {stats['anomaly_count']}")
    print(f"Normal violations: {stats['normal_violations']}")

    assert stats['total_violations'] == 5
    assert stats['anomaly_count'] == 2
    assert stats['normal_violations'] == 3
    print("[PASS] Statistics tracking works correctly")

    return True


def test_anomaly_descriptions():
    """测试异常原因描述"""
    print("\n" + "=" * 50)
    print("Test 4: Anomaly Descriptions")
    print("=" * 50)

    for reason, desc in ANOMALY_DESCRIPTIONS.items():
        print(f"  {reason.value}: {desc}")

    assert len(ANOMALY_DESCRIPTIONS) == 4
    print("[PASS] All anomaly descriptions available")

    return True


def test_snapshot_naming():
    """测试截图命名"""
    print("\n" + "=" * 50)
    print("Test 5: Snapshot Naming Convention")
    print("=" * 50)

    from datetime import datetime

    timestamp = datetime.now()
    record_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

    print(f"  Sample record ID: {record_id}")
    print(f"  Format: YYYYMMDD_HHMMSS_ffffff")

    parts = record_id.split('_')
    assert len(parts) == 3
    assert len(parts[0]) == 8
    assert len(parts[1]) == 6
    assert len(parts[2]) == 6

    print("[PASS] Snapshot naming convention is correct")

    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("ADAPTIVE VIOLATION DETECTION SYSTEM - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Emergency Vehicle Detector", test_emergency_vehicle_detector),
        ("Adaptive Violation Detector", test_adaptive_violation_detector),
        ("Statistics Tracking", test_statistics),
        ("Anomaly Descriptions", test_anomaly_descriptions),
        ("Snapshot Naming", test_snapshot_naming),
    ]

    try:
        from test_lane_violation import test_wrong_way, test_illegal_lane_change
        tests.append(("Wrong Way Detection", test_wrong_way))
        tests.append(("Illegal Lane Change", test_illegal_lane_change))
    except ImportError:
        pass

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[FAILURE] Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
