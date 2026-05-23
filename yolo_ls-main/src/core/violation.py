"""违规检测模块：闯红灯、超速"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque


class ViolationType(Enum):
    """违规类型"""
    RED_LIGHT = "red_light"  # 闯红灯
    SPEEDING = "speeding"    # 超速


@dataclass
class Violation:
    """违规记录"""
    violation_type: ViolationType
    track_id: int
    timestamp: datetime
    location: Tuple[int, int]  # 违规位置
    speed: Optional[float] = None  # 超速时的速度
    plate_number: Optional[str] = None
    snapshot: Optional[np.ndarray] = None  # 违规截图


class TrafficLightDetector:
    """交通灯状态检测器"""

    def __init__(self):
        # 红灯 HSV 范围
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        # 绿灯 HSV 范围
        self.green_lower = np.array([35, 100, 100])
        self.green_upper = np.array([85, 255, 255])
        # 黄灯 HSV 范围
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])

    def detect_state(self, frame: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> str:
        """
        检测交通灯状态

        Args:
            frame: BGR 图像
            bbox: 交通灯边界框

        Returns:
            状态: 'red', 'green', 'yellow', 'unknown'
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return 'unknown'

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 计算各颜色像素比例
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = red_mask1 | red_mask2
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)

        total = roi.shape[0] * roi.shape[1]
        red_ratio = np.sum(red_mask > 0) / total
        green_ratio = np.sum(green_mask > 0) / total
        yellow_ratio = np.sum(yellow_mask > 0) / total

        # 判断状态
        threshold = 0.1
        if red_ratio > threshold and red_ratio > green_ratio and red_ratio > yellow_ratio:
            return 'red'
        elif green_ratio > threshold and green_ratio > red_ratio and green_ratio > yellow_ratio:
            return 'green'
        elif yellow_ratio > threshold:
            return 'yellow'

        return 'unknown'


class StopLine:
    """停止线"""

    def __init__(self, y: int, x_start: int, x_end: int):
        """
        初始化停止线

        Args:
            y: 停止线 y 坐标
            x_start: 停止线起始 x 坐标
            x_end: 停止线结束 x 坐标
        """
        self.y = y
        self.x_start = x_start
        self.x_end = x_end

    def is_crossed(self, prev_center: Tuple[int, int],
                   curr_center: Tuple[int, int]) -> bool:
        """
        判断是否越过停止线

        Args:
            prev_center: 前一帧中心点
            curr_center: 当前帧中心点

        Returns:
            是否越线
        """
        # 检查 x 是否在停止线范围内
        if not (self.x_start <= curr_center[0] <= self.x_end):
            return False

        # 检查是否从上方越过停止线
        if prev_center[1] < self.y <= curr_center[1]:
            return True

        return False


class ViolationDetector:
    """违规检测器"""

    def __init__(self, speed_limit: float = 60.0,
                 stop_line: Optional[StopLine] = None):
        """
        初始化违规检测器

        Args:
            speed_limit: 速度限制 (km/h)
            stop_line: 停止线对象
        """
        self.speed_limit = speed_limit
        self.stop_line = stop_line
        self.traffic_light_detector = TrafficLightDetector()

        # 跟踪历史
        self.track_history: Dict[int, deque] = {}
        # 已记录的违规（避免重复）
        self.recorded_violations: Dict[int, set] = {}
        # 当前交通灯状态
        self.current_light_state = 'unknown'

    def set_stop_line(self, y: int, x_start: int, x_end: int):
        """设置停止线"""
        self.stop_line = StopLine(y, x_start, x_end)

    def update_traffic_light(self, frame: np.ndarray,
                             light_bbox: Optional[Tuple[int, int, int, int]]):
        """更新交通灯状态"""
        if light_bbox:
            self.current_light_state = self.traffic_light_detector.detect_state(
                frame, light_bbox
            )

    def check_violations(self, track_id: int, center: Tuple[int, int],
                         speed: float, frame: np.ndarray) -> List[Violation]:
        """
        检查违规行为

        Args:
            track_id: 跟踪 ID
            center: 当前中心点
            speed: 当前速度
            frame: 当前帧

        Returns:
            违规列表
        """
        violations = []

        # 初始化跟踪历史
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=10)
            self.recorded_violations[track_id] = set()

        history = self.track_history[track_id]

        # 检查超速
        if speed > self.speed_limit:
            if ViolationType.SPEEDING not in self.recorded_violations[track_id]:
                violations.append(Violation(
                    violation_type=ViolationType.SPEEDING,
                    track_id=track_id,
                    timestamp=datetime.now(),
                    location=center,
                    speed=speed,
                    snapshot=frame.copy()
                ))
                self.recorded_violations[track_id].add(ViolationType.SPEEDING)

        # 检查闯红灯
        if (self.stop_line and len(history) > 0 and
                self.current_light_state == 'red'):
            prev_center = history[-1]
            if self.stop_line.is_crossed(prev_center, center):
                if ViolationType.RED_LIGHT not in self.recorded_violations[track_id]:
                    violations.append(Violation(
                        violation_type=ViolationType.RED_LIGHT,
                        track_id=track_id,
                        timestamp=datetime.now(),
                        location=center,
                        snapshot=frame.copy()
                    ))
                    self.recorded_violations[track_id].add(ViolationType.RED_LIGHT)

        # 更新历史
        history.append(center)

        return violations

    def clear_track(self, track_id: int):
        """清除跟踪记录"""
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.recorded_violations:
            del self.recorded_violations[track_id]

    def draw_stop_line(self, frame: np.ndarray,
                       color: Tuple[int, int, int] = (0, 0, 255),
                       thickness: int = 2) -> np.ndarray:
        """在帧上绘制停止线"""
        if self.stop_line:
            cv2.line(
                frame,
                (self.stop_line.x_start, self.stop_line.y),
                (self.stop_line.x_end, self.stop_line.y),
                color, thickness
            )
        return frame
