"""特征分析模块：颜色、速度、方向"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    """方向枚举"""
    UNKNOWN = "unknown"
    NORTH = "north"      # 上
    SOUTH = "south"      # 下
    EAST = "east"        # 右
    WEST = "west"        # 左
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"


# 颜色名称映射
COLOR_NAMES = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'yellow': (0, 255, 255),
    'silver': (192, 192, 192),
    'gray': (128, 128, 128),
    'orange': (0, 165, 255),
    'brown': (42, 42, 165),
}

# HSV 颜色范围
HSV_COLOR_RANGES = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'red2': [(160, 100, 100), (180, 255, 255)],
    'orange': [(11, 100, 100), (25, 255, 255)],
    'yellow': [(26, 100, 100), (35, 255, 255)],
    'green': [(36, 100, 100), (85, 255, 255)],
    'blue': [(86, 100, 100), (125, 255, 255)],
    'white': [(0, 0, 200), (180, 30, 255)],
    'black': [(0, 0, 0), (180, 255, 50)],
    'gray': [(0, 0, 51), (180, 50, 199)],
    'silver': [(0, 0, 150), (180, 30, 220)],
}


@dataclass
class VehicleFeatures:
    """车辆特征数据类"""
    track_id: int
    color: str
    color_confidence: float
    speed: float  # km/h
    direction: Direction
    bbox: Tuple[int, int, int, int]


class ColorAnalyzer:
    """颜色分析器（默认使用光照鲁棒检测）"""

    def __init__(self, use_robust: bool = True):
        self.use_robust = use_robust
        self.color_ranges = HSV_COLOR_RANGES
        self._robust = None
        if use_robust:
            from .robust_color import RobustColorDetector
            self._robust = RobustColorDetector()

    def analyze(self, frame: np.ndarray,
                bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        if self._robust is not None:
            return self._robust.detect(frame, bbox, light_condition='auto')
        return self._analyze_legacy(frame, bbox)

    def _analyze_legacy(self, frame: np.ndarray,
                        bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        x1, y1, x2, y2 = bbox
        margin_x = (x2 - x1) // 4
        margin_y = (y2 - y1) // 4
        roi = frame[y1 + margin_y:y2 - margin_y, x1 + margin_x:x2 - margin_x]
        if roi.size == 0:
            return 'unknown', 0.0
        from src.utils.hsv import bgr_to_hsv_normalized
        hsv = bgr_to_hsv_normalized(roi)
        color_scores = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]
        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            score = np.sum(mask > 0) / total_pixels
            if color_name == 'red2':
                color_scores['red'] = color_scores.get('red', 0) + score
            else:
                color_scores[color_name] = color_scores.get(color_name, 0) + score
        if not color_scores:
            return 'unknown', 0.0
        best_color = max(color_scores, key=color_scores.get)
        return best_color, color_scores[best_color]


class SpeedCalculator:
    """速度计算器"""

    def __init__(self, pixel_to_meter: float = 0.05, fps: float = 15):
        """
        初始化速度计算器

        Args:
            pixel_to_meter: 像素到米的转换系数
            fps: 视频帧率
        """
        self.pixel_to_meter = pixel_to_meter
        self.fps = fps
        self.track_history: Dict[int, deque] = {}

    def update(self, track_id: int, center: Tuple[int, int]) -> float:
        """
        更新跟踪点并计算速度

        Args:
            track_id: 跟踪 ID
            center: 当前中心点

        Returns:
            速度 (km/h)
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=10)

        history = self.track_history[track_id]
        history.append(center)

        if len(history) < 2:
            return 0.0

        # 计算最近几帧的平均速度
        total_dist = 0.0
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]
            total_dist += np.sqrt(dx ** 2 + dy ** 2)

        avg_pixel_speed = total_dist / (len(history) - 1)  # 像素/帧
        meter_per_second = avg_pixel_speed * self.pixel_to_meter * self.fps
        km_per_hour = meter_per_second * 3.6

        return km_per_hour

    def get_direction(self, track_id: int) -> Direction:
        """
        获取运动方向

        Args:
            track_id: 跟踪 ID

        Returns:
            方向枚举
        """
        if track_id not in self.track_history:
            return Direction.UNKNOWN

        history = self.track_history[track_id]
        if len(history) < 2:
            return Direction.UNKNOWN

        # 计算总位移
        dx = history[-1][0] - history[0][0]
        dy = history[-1][1] - history[0][1]

        if abs(dx) < 5 and abs(dy) < 5:
            return Direction.UNKNOWN

        angle = np.arctan2(-dy, dx) * 180 / np.pi  # 注意 y 轴向下

        # 根据角度判断方向
        if -22.5 <= angle < 22.5:
            return Direction.EAST
        elif 22.5 <= angle < 67.5:
            return Direction.NORTHEAST
        elif 67.5 <= angle < 112.5:
            return Direction.NORTH
        elif 112.5 <= angle < 157.5:
            return Direction.NORTHWEST
        elif angle >= 157.5 or angle < -157.5:
            return Direction.WEST
        elif -157.5 <= angle < -112.5:
            return Direction.SOUTHWEST
        elif -112.5 <= angle < -67.5:
            return Direction.SOUTH
        elif -67.5 <= angle < -22.5:
            return Direction.SOUTHEAST

        return Direction.UNKNOWN

    def clear(self, track_id: int):
        """清除指定跟踪的历史"""
        if track_id in self.track_history:
            del self.track_history[track_id]


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, pixel_to_meter: float = 0.05, fps: float = 15):
        self.color_analyzer = ColorAnalyzer()
        self.speed_calculator = SpeedCalculator(pixel_to_meter, fps)

    def extract(self, frame: np.ndarray, track_id: int,
                bbox: Tuple[int, int, int, int]) -> VehicleFeatures:
        """
        提取车辆特征

        Args:
            frame: BGR 图像
            track_id: 跟踪 ID
            bbox: 边界框

        Returns:
            VehicleFeatures 对象
        """
        # 颜色分析
        color, color_conf = self.color_analyzer.analyze(frame, bbox)

        # 计算中心点
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        # 速度计算
        speed = self.speed_calculator.update(track_id, center)

        # 方向判断
        direction = self.speed_calculator.get_direction(track_id)

        return VehicleFeatures(
            track_id=track_id,
            color=color,
            color_confidence=color_conf,
            speed=speed,
            direction=direction,
            bbox=bbox
        )
