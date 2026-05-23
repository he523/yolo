"""光照鲁棒的颜色检测（多时段 HSV 范围 + 自动光照估计）。"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.hsv import bgr_to_hsv_normalized

# 多光照条件 HSV 范围 (H, S, V)
MULTI_LIGHT_RANGES: Dict[str, List[Dict[str, Tuple]]] = {
    'red': [
        {'lower': (0, 100, 80), 'upper': (10, 255, 255), 'condition': 'day'},
        {'lower': (0, 80, 60), 'upper': (10, 255, 220), 'condition': 'night'},
        {'lower': (160, 80, 60), 'upper': (180, 255, 220), 'condition': 'night'},
    ],
    'blue': [
        {'lower': (86, 100, 80), 'upper': (125, 255, 255), 'condition': 'day'},
        {'lower': (86, 60, 50), 'upper': (125, 255, 200), 'condition': 'night'},
    ],
    'white': [
        {'lower': (0, 0, 200), 'upper': (180, 30, 255), 'condition': 'day'},
        {'lower': (0, 0, 160), 'upper': (180, 40, 255), 'condition': 'night'},
    ],
    'black': [
        {'lower': (0, 0, 0), 'upper': (180, 255, 50), 'condition': 'day'},
        {'lower': (0, 0, 0), 'upper': (180, 255, 40), 'condition': 'night'},
    ],
    'yellow': [
        {'lower': (26, 100, 100), 'upper': (35, 255, 255), 'condition': 'day'},
        {'lower': (20, 80, 80), 'upper': (40, 255, 230), 'condition': 'night'},
    ],
    'green': [
        {'lower': (36, 100, 100), 'upper': (85, 255, 255), 'condition': 'day'},
        {'lower': (36, 60, 60), 'upper': (85, 255, 200), 'condition': 'night'},
    ],
}


class RobustColorDetector:
    """根据光照条件选择 HSV 阈值进行车身颜色分析。"""

    def __init__(self, color_ranges: Optional[Dict] = None):
        self.color_ranges = color_ranges or MULTI_LIGHT_RANGES

    @staticmethod
    def estimate_light_condition(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> str:
        """根据 ROI 平均亮度估计 day / night。"""
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = frame[max(0, y1):y2, max(0, x1):x2]
        else:
            roi = frame
        if roi.size == 0:
            return 'day'
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        mean_v = float(np.mean(gray))
        return 'night' if mean_v < 85 else 'day'

    def detect(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        light_condition: str = 'auto',
    ) -> Tuple[str, float]:
        x1, y1, x2, y2 = bbox
        margin_x = (x2 - x1) // 4
        margin_y = (y2 - y1) // 4
        roi = frame[y1 + margin_y:y2 - margin_y, x1 + margin_x:x2 - margin_x]
        if roi.size == 0:
            return 'unknown', 0.0

        if light_condition == 'auto':
            light_condition = self.estimate_light_condition(frame, bbox)

        hsv = bgr_to_hsv_normalized(roi)
        total = hsv.shape[0] * hsv.shape[1]
        color_scores: Dict[str, float] = {}

        for color_name, ranges in self.color_ranges.items():
            score = 0.0
            for entry in ranges:
                if entry.get('condition', 'day') != light_condition:
                    continue
                lower, upper = entry['lower'], entry['upper']
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                score = max(score, np.sum(mask > 0) / total)
            color_scores[color_name] = score

        if not color_scores:
            return 'unknown', 0.0
        best = max(color_scores, key=color_scores.get)
        return best, color_scores[best]
