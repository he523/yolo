"""帧预处理模块"""
import cv2
import numpy as np
from typing import Tuple


class FramePreprocessor:
    """帧预处理器"""

    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        初始化预处理器

        Args:
            target_size: 目标尺寸 (width, height)
        """
        self.target_size = target_size

    def resize(self, frame: np.ndarray) -> np.ndarray:
        """调整帧尺寸"""
        return cv2.resize(frame, self.target_size)

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """归一化像素值到 [0, 1]"""
        return frame.astype(np.float32) / 255.0

    def to_rgb(self, frame: np.ndarray) -> np.ndarray:
        """BGR 转 RGB"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def preprocess(self, frame: np.ndarray, resize: bool = True,
                   normalize: bool = False, to_rgb: bool = False) -> np.ndarray:
        """
        完整预处理流程

        Args:
            frame: 输入帧
            resize: 是否调整尺寸
            normalize: 是否归一化
            to_rgb: 是否转换为RGB

        Returns:
            预处理后的帧
        """
        result = frame.copy()
        if resize:
            result = self.resize(result)
        if to_rgb:
            result = self.to_rgb(result)
        if normalize:
            result = self.normalize(result)
        return result

    def is_valid_frame(self, frame: np.ndarray) -> bool:
        """检查帧是否有效"""
        if frame is None:
            return False
        if frame.size == 0:
            return False
        # 检查是否全黑或全白
        mean_val = np.mean(frame)
        if mean_val < 5 or mean_val > 250:
            return False
        return True
