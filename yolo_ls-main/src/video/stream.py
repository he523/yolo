"""视频流接入模块"""
import cv2
import numpy as np
from typing import Generator, Optional, Tuple
from pathlib import Path


class VideoStream:
    """视频流处理类，支持摄像头、RTSP流、本地视频文件"""

    def __init__(self, source: str = "0", fps: int = 15,
                 width: int = 1280, height: int = 720):
        """
        初始化视频流

        Args:
            source: 视频源（摄像头ID、RTSP地址或视频文件路径）
            fps: 目标帧率
            width: 视频宽度
            height: 视频高度
        """
        self.source = int(source) if source.isdigit() else source
        self.target_fps = fps
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.is_file = False

    def open(self) -> bool:
        """打开视频流"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        # 判断是否为文件
        if isinstance(self.source, str) and Path(self.source).exists():
            self.is_file = True

        # 设置摄像头参数
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧"""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
        return ret, frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """帧生成器"""
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame

    def get_fps(self) -> float:
        """获取实际帧率"""
        if self.cap:
            return self.cap.get(cv2.CAP_PROP_FPS)
        return self.target_fps

    def get_frame_size(self) -> Tuple[int, int]:
        """获取帧尺寸 (width, height)"""
        if self.cap:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return w, h
        return self.width, self.height

    def get_total_frames(self) -> int:
        """获取总帧数（仅视频文件有效）"""
        if self.cap and self.is_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
