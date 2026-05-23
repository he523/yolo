#!/usr/bin/env python3
"""逆行与违规变道核心算法单元测试（unittest，兼容 pytest 发现）。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.lane_violation import (
    LaneViolationAnalyzer,
    WrongWayConfig,
    IllegalLaneChangeConfig,
)
from src.core.adaptive_violation import AdaptiveViolationDetector, ViolationType
from src.core.feature import Direction


def _trajectory_south_forward(start_y: int = 100, steps: int = 8, step: int = 12):
    return [(320, start_y + i * step) for i in range(steps)]


def _trajectory_south_backward(start_y: int = 200, steps: int = 8, step: int = 12):
    return [(320, start_y - i * step) for i in range(steps)]


def _trajectory_lane_change_south(start_x: int = 200, steps: int = 10, step_y: int = 10):
    pts = []
    for i in range(steps // 2):
        pts.append((start_x, 100 + i * step_y))
    for i in range(steps // 2, steps):
        pts.append((start_x + 90, 100 + i * step_y))
    return pts


class TestWrongWayCore(unittest.TestCase):
    def test_wrong_way_opposing_trajectory(self):
        analyzer = LaneViolationAnalyzer(
            wrong_way_config=WrongWayConfig(expected_flow_direction='south'),
        )
        detected, conf = analyzer.detect_wrong_way(_trajectory_south_backward(), speed_kmh=25.0)
        self.assertTrue(detected)
        self.assertGreater(conf, 0.5)

    def test_wrong_way_legal_trajectory_not_flagged(self):
        analyzer = LaneViolationAnalyzer(
            wrong_way_config=WrongWayConfig(expected_flow_direction='south'),
        )
        detected, _ = analyzer.detect_wrong_way(_trajectory_south_forward(), speed_kmh=25.0)
        self.assertFalse(detected)

    def test_wrong_way_low_speed_ignored(self):
        analyzer = LaneViolationAnalyzer(
            wrong_way_config=WrongWayConfig(
                expected_flow_direction='south',
                min_speed_kmh=20.0,
            ),
        )
        detected, _ = analyzer.detect_wrong_way(_trajectory_south_backward(), speed_kmh=5.0)
        self.assertFalse(detected)

    def test_wrong_way_angle_method(self):
        analyzer = LaneViolationAnalyzer(
            wrong_way_config=WrongWayConfig(
                expected_flow_direction='south',
                direction_threshold_deg=30,
            ),
        )
        centers = [(320, 200 - i * 15) for i in range(8)]
        detected, conf = analyzer.detect_wrong_way(centers, speed_kmh=25.0)
        self.assertTrue(detected)
        self.assertGreater(conf, 0.4)

    def test_wrong_way_direction_enum_fallback(self):
        analyzer = LaneViolationAnalyzer(
            wrong_way_config=WrongWayConfig(expected_flow_direction='south'),
        )
        detected, conf = analyzer.detect_wrong_way(
            [(100, 100)],
            speed_kmh=30.0,
            direction=Direction.NORTH,
        )
        self.assertTrue(detected)
        self.assertGreaterEqual(conf, 0.5)


class TestIllegalLaneChangeCore(unittest.TestCase):
    def test_illegal_lane_change_detected(self):
        analyzer = LaneViolationAnalyzer(
            lane_change_config=IllegalLaneChangeConfig(
                expected_flow_direction='south',
                lateral_shift_px=70.0,
                max_lateral_jump_px=30.0,
            ),
        )
        detected, conf = analyzer.detect_illegal_lane_change(
            _trajectory_lane_change_south(), speed_kmh=30.0,
        )
        self.assertTrue(detected)
        self.assertGreater(conf, 0.4)

    def test_straight_line_not_lane_change(self):
        analyzer = LaneViolationAnalyzer(
            lane_change_config=IllegalLaneChangeConfig(expected_flow_direction='south'),
        )
        detected, _ = analyzer.detect_illegal_lane_change(
            _trajectory_south_forward(), speed_kmh=30.0,
        )
        self.assertFalse(detected)

    def test_wrong_way_not_classified_as_lane_change(self):
        analyzer = LaneViolationAnalyzer(
            wrong_way_config=WrongWayConfig(expected_flow_direction='south'),
            lane_change_config=IllegalLaneChangeConfig(expected_flow_direction='south'),
        )
        result = analyzer.analyze(
            _trajectory_south_backward(),
            speed_kmh=30.0,
            wrong_way_enabled=True,
            illegal_lane_enabled=True,
        )
        self.assertTrue(result.is_wrong_way)
        self.assertFalse(result.is_illegal_lane_change)


class TestAdaptiveViolationIntegration(unittest.TestCase):
    def test_adaptive_detector_wrong_way_violation(self):
        detector = AdaptiveViolationDetector(
            speed_limit=120,
            snapshot_dir='data/test_snapshots',
            expected_flow_direction='south',
            wrong_way_enabled=True,
            illegal_lane_enabled=False,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        record = None
        for center in _trajectory_south_backward():
            r = detector.check_violation(
                track_id=101,
                bbox=(center[0] - 40, center[1] - 30, center[0] + 40, center[1] + 30),
                speed=35.0,
                frame=frame,
                direction=Direction.NORTH,
            )
            if r is not None:
                record = r
        self.assertIsNotNone(record)
        self.assertEqual(record.violation_type, ViolationType.WRONG_WAY)

    def test_adaptive_detector_illegal_lane_violation(self):
        detector = AdaptiveViolationDetector(
            speed_limit=120,
            snapshot_dir='data/test_snapshots',
            expected_flow_direction='south',
            wrong_way_enabled=False,
            illegal_lane_enabled=True,
            lane_change_lateral_px=70,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        record = None
        for center in _trajectory_lane_change_south():
            r = detector.check_violation(
                track_id=102,
                bbox=(center[0] - 40, center[1] - 25, center[0] + 40, center[1] + 25),
                speed=40.0,
                frame=frame,
                direction=Direction.SOUTH,
            )
            if r is not None:
                record = r
        self.assertIsNotNone(record)
        self.assertEqual(record.violation_type, ViolationType.ILLEGAL_LANE)

    def test_wrong_way_disabled(self):
        detector = AdaptiveViolationDetector(
            speed_limit=120,
            snapshot_dir='data/test_snapshots',
            expected_flow_direction='south',
            wrong_way_enabled=False,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        record = None
        for center in _trajectory_south_backward():
            record = detector.check_violation(
                track_id=200,
                bbox=(center[0] - 40, center[1] - 30, center[0] + 40, center[1] + 30),
                speed=35.0,
                frame=frame,
                direction=Direction.NORTH,
            )
        if record is not None:
            self.assertNotEqual(record.violation_type, ViolationType.WRONG_WAY)


def test_wrong_way():
    """pytest 风格入口：逆行检测。"""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestWrongWayCore))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    assert result.wasSuccessful()


def test_illegal_lane_change():
    """pytest 风格入口：违规变道检测。"""
    suite = unittest.TestSuite()
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIllegalLaneChangeCore))
    result = unittest.TextTestRunner(verbosity=0).run(suite)
    assert result.wasSuccessful()


if __name__ == '__main__':
    unittest.main(verbosity=2)
