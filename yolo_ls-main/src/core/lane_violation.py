"""
逆行与违规变道核心算法。

基于轨迹位移向量与期望车流方向，避免单帧方向枚举误判。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .feature import Direction
from src.utils.constants import (
    OPPOSITE_DIRECTIONS,
    DEFAULT_LANE_CHANGE_LATERAL_PX,
    DEFAULT_LANE_CHANGE_MIN_SPEED_KMH,
    DEFAULT_LANE_CHANGE_HISTORY_LEN,
)

__all__ = [
    'LaneViolationAnalyzer',
    'LaneViolationResult',
    'WrongWayConfig',
    'IllegalLaneChangeConfig',
    'build_analyzer_from_violation_config',
    'FLOW_UNIT_VECTORS',
]

# 图像坐标系：y 向下为正。期望车流方向的单位向量 (dx, dy)
FLOW_UNIT_VECTORS: Dict[str, Tuple[float, float]] = {
    'north': (0.0, -1.0),
    'south': (0.0, 1.0),
    'east': (1.0, 0.0),
    'west': (-1.0, 0.0),
    'northeast': (1.0, -1.0),
    'northwest': (-1.0, -1.0),
    'southeast': (1.0, 1.0),
    'southwest': (-1.0, 1.0),
}


def _normalize(v: Tuple[float, float]) -> Tuple[float, float]:
    mag = math.hypot(v[0], v[1])
    if mag < 1e-6:
        return 0.0, 0.0
    return v[0] / mag, v[1] / mag


def _flow_vector(expected_flow: str) -> Optional[Tuple[float, float]]:
    key = expected_flow.lower().strip()
    raw = FLOW_UNIT_VECTORS.get(key)
    if raw is None:
        return None
    return _normalize(raw)


def _lateral_vector(flow: Tuple[float, float]) -> Tuple[float, float]:
    """与车流垂直的单位向量（图像坐标）。"""
    return _normalize((-flow[1], flow[0]))


def _project_scalar(point: Tuple[int, int], axis: Tuple[float, float]) -> float:
    return point[0] * axis[0] + point[1] * axis[1]


def _flow_angle_deg(expected_flow: str) -> Optional[float]:
    """道路主流向角度（度），图像坐标 atan2(dy, dx)。"""
    flow = _flow_vector(expected_flow)
    if flow is None:
        return None
    return math.degrees(math.atan2(flow[1], flow[0]))


def _angle_difference_deg(a: float, b: float) -> float:
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


@dataclass
class WrongWayConfig:
    expected_flow_direction: str = 'south'
    min_speed_kmh: float = DEFAULT_LANE_CHANGE_MIN_SPEED_KMH
    min_net_displacement_px: float = 40.0
    min_step_displacement_px: float = 3.0
    opposing_step_ratio: float = 0.65
    direction_threshold_deg: float = 30.0
    min_points_for_angle: int = 5


@dataclass
class IllegalLaneChangeConfig:
    expected_flow_direction: str = 'south'
    min_speed_kmh: float = DEFAULT_LANE_CHANGE_MIN_SPEED_KMH
    lateral_shift_px: float = float(DEFAULT_LANE_CHANGE_LATERAL_PX)
    max_lateral_jump_px: float = 35.0
    min_forward_displacement_px: float = 25.0
    min_history_points: int = 5


@dataclass
class LaneViolationResult:
    is_wrong_way: bool = False
    is_illegal_lane_change: bool = False
    wrong_way_confidence: float = 0.0
    lane_change_confidence: float = 0.0


class LaneViolationAnalyzer:
    """逆行 / 违规变道分析器（无状态，可单测）。"""

    def __init__(
        self,
        wrong_way_config: Optional[WrongWayConfig] = None,
        lane_change_config: Optional[IllegalLaneChangeConfig] = None,
    ):
        self.wrong_way_config = wrong_way_config or WrongWayConfig()
        self.lane_change_config = lane_change_config or IllegalLaneChangeConfig()

    def analyze(
        self,
        centers: Sequence[Tuple[int, int]],
        speed_kmh: float,
        direction: Optional[Direction] = None,
        wrong_way_enabled: bool = True,
        illegal_lane_enabled: bool = True,
    ) -> LaneViolationResult:
        result = LaneViolationResult()
        if not centers:
            return result

        if wrong_way_enabled:
            ww, conf = self.detect_wrong_way(centers, speed_kmh, direction)
            result.is_wrong_way = ww
            result.wrong_way_confidence = conf

        if illegal_lane_enabled and not result.is_wrong_way:
            ilc, conf = self.detect_illegal_lane_change(centers, speed_kmh)
            result.is_illegal_lane_change = ilc
            result.lane_change_confidence = conf

        return result

    def detect_wrong_way(
        self,
        centers: Sequence[Tuple[int, int]],
        speed_kmh: float,
        direction: Optional[Direction] = None,
    ) -> Tuple[bool, float]:
        """
        逆行判定：
        1) 连续多帧位移与期望车流点积为负的比例 ≥ 阈值；
        2) 净位移沿车流反向且幅度足够；
        3) 可选：方向枚举与期望相反作为辅助（历史不足时）。
        """
        cfg = self.wrong_way_config
        if speed_kmh < cfg.min_speed_kmh:
            return False, 0.0

        flow = _flow_vector(cfg.expected_flow_direction)
        if flow is None:
            return False, 0.0

        points = list(centers)
        if len(points) >= 2:
            opposing, total = self._count_opposing_steps(points, flow, cfg.min_step_displacement_px)
            if total >= 2:
                step_ratio = opposing / total
                net_dx = points[-1][0] - points[0][0]
                net_dy = points[-1][1] - points[0][1]
                net_mag = math.hypot(net_dx, net_dy)
                net_dot = net_dx * flow[0] + net_dy * flow[1]
                if (
                    step_ratio >= cfg.opposing_step_ratio
                    and net_dot < 0
                    and net_mag >= cfg.min_net_displacement_px
                ):
                    confidence = min(1.0, step_ratio * 0.6 + min(1.0, net_mag / 120.0) * 0.4)
                    return True, confidence

        angle_hit, angle_conf = self._detect_wrong_way_by_angle(points, speed_kmh)
        if angle_hit:
            return True, angle_conf

        if direction is not None and direction != Direction.UNKNOWN:
            opposite = OPPOSITE_DIRECTIONS.get(cfg.expected_flow_direction.lower())
            if opposite and direction.value == opposite:
                return True, 0.55

        return False, 0.0

    def _detect_wrong_way_by_angle(
        self,
        points: Sequence[Tuple[int, int]],
        speed_kmh: float,
    ) -> Tuple[bool, float]:
        """基于行驶方向与道路主方向夹角判定逆行（建议中的角度法）。"""
        cfg = self.wrong_way_config
        if speed_kmh < cfg.min_speed_kmh or len(points) < cfg.min_points_for_angle:
            return False, 0.0
        road_angle = _flow_angle_deg(cfg.expected_flow_direction)
        if road_angle is None:
            return False, 0.0
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        if math.hypot(dx, dy) < cfg.min_net_displacement_px * 0.5:
            return False, 0.0
        vehicle_angle = math.degrees(math.atan2(dy, dx))
        diff = _angle_difference_deg(vehicle_angle, road_angle)
        threshold = cfg.direction_threshold_deg
        if diff > (180.0 - threshold):
            conf = min(1.0, (diff - (180.0 - threshold)) / max(threshold, 1.0))
            return True, max(0.5, conf)
        return False, 0.0

    def detect_illegal_lane_change(
        self,
        centers: Sequence[Tuple[int, int]],
        speed_kmh: float,
    ) -> Tuple[bool, float]:
        """
        违规变道判定：
        1) 轨迹在横向轴上出现明显阶跃（前后半段横向均值差或单帧大跳变）；
        2) 同期仍有足够纵向（沿车流）位移，排除原地掉头/停车；
        3) 净位移仍大致沿期望车流方向（与逆行区分）。
        """
        cfg = self.lane_change_config
        if speed_kmh < cfg.min_speed_kmh:
            return False, 0.0

        flow = _flow_vector(cfg.expected_flow_direction)
        if flow is None:
            return False, 0.0

        points = list(centers)
        if len(points) < cfg.min_history_points:
            return False, 0.0

        lateral_axis = _lateral_vector(flow)
        lateral_vals = [_project_scalar(p, lateral_axis) for p in points]
        forward_vals = [_project_scalar(p, flow) for p in points]

        forward_disp = forward_vals[-1] - forward_vals[0]
        if abs(forward_disp) < cfg.min_forward_displacement_px:
            return False, 0.0

        # 净位移须与车流同向（排除掉头误判为变道）
        if forward_disp <= 0:
            return False, 0.0

        mid = len(points) // 2
        first_mean = sum(lateral_vals[:mid]) / max(1, mid)
        second_mean = sum(lateral_vals[mid:]) / max(1, len(points) - mid)
        lateral_shift = abs(second_mean - first_mean)

        max_jump = 0.0
        for i in range(1, len(lateral_vals)):
            max_jump = max(max_jump, abs(lateral_vals[i] - lateral_vals[i - 1]))

        shift_ok = lateral_shift >= cfg.lateral_shift_px
        jump_ok = max_jump >= cfg.max_lateral_jump_px

        if not (shift_ok or jump_ok):
            return False, 0.0

        # 变道时横向跨度应明显但小于“掉头级”横向/纵向比
        lateral_span = max(lateral_vals) - min(lateral_vals)
        if lateral_span < cfg.lateral_shift_px * 0.5:
            return False, 0.0
        if lateral_span > abs(forward_disp) * 2.5:
            return False, 0.0

        confidence = min(
            1.0,
            (lateral_shift / max(cfg.lateral_shift_px, 1)) * 0.5
            + (max_jump / max(cfg.max_lateral_jump_px, 1)) * 0.5,
        )
        return True, confidence

    @staticmethod
    def _count_opposing_steps(
        points: Sequence[Tuple[int, int]],
        flow: Tuple[float, float],
        min_step_px: float,
    ) -> Tuple[int, int]:
        opposing = 0
        total = 0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i - 1][0]
            dy = points[i][1] - points[i - 1][1]
            mag = math.hypot(dx, dy)
            if mag < min_step_px:
                continue
            total += 1
            if dx * flow[0] + dy * flow[1] < 0:
                opposing += 1
        return opposing, total


def build_analyzer_from_violation_config(
    expected_flow_direction: str = 'south',
    lane_change_lateral_px: int = DEFAULT_LANE_CHANGE_LATERAL_PX,
    lane_change_min_speed_kmh: float = DEFAULT_LANE_CHANGE_MIN_SPEED_KMH,
    history_len: int = DEFAULT_LANE_CHANGE_HISTORY_LEN,
) -> LaneViolationAnalyzer:
    """从 AdaptiveViolationDetector 参数构建分析器。"""
    return LaneViolationAnalyzer(
        wrong_way_config=WrongWayConfig(
            expected_flow_direction=expected_flow_direction,
            min_speed_kmh=lane_change_min_speed_kmh,
        ),
        lane_change_config=IllegalLaneChangeConfig(
            expected_flow_direction=expected_flow_direction,
            min_speed_kmh=lane_change_min_speed_kmh,
            lateral_shift_px=float(lane_change_lateral_px),
            min_history_points=max(5, history_len // 2),
        ),
    )
