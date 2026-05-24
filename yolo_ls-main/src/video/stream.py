"""视频流接入模块"""
from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

VideoSource = Union[int, str]


class VideoStream:
    """视频流处理类，支持摄像头、RTSP流、本地视频文件"""

    def __init__(
        self,
        source: str = "0",
        fps: int = 15,
        width: int = 1280,
        height: int = 720,
        resize: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        初始化视频流

        Args:
            source: 视频源（摄像头ID、RTSP地址或视频文件路径）
            fps: 目标帧率
            width: 视频宽度
            height: 视频高度
            resize: 可选的目标尺寸 (w, h)，读帧时缩放
        """
        self.source: VideoSource = int(source) if str(source).isdigit() else source
        self.target_fps = fps
        self.width = width
        self.height = height
        self.resize_dims: Optional[Tuple[int, int]] = (
            tuple(int(v) for v in resize) if resize else None
        )
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.is_file = False

    def open(self) -> bool:
        """打开视频流"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        if isinstance(self.source, str) and Path(self.source).exists():
            self.is_file = True

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        return True

    def grab(self) -> bool:
        """仅抓取下一帧不解码（用于跳帧时降低开销）。"""
        if self.cap is None:
            return False
        return bool(self.cap.grab())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """读取一帧"""
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret and self.resize_dims is not None:
            frame = cv2.resize(frame, self.resize_dims, interpolation=cv2.INTER_LINEAR)
        if ret:
            self.frame_count += 1
        return ret, frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """帧生成器"""
        while True:
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield frame

    def get_fps(self) -> float:
        """获取实际帧率；无效时回退 target_fps。"""
        if self.cap is None:
            return float(self.target_fps)
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            return float(self.target_fps)
        return fps

    def get_frame_size(self) -> Tuple[int, int]:
        """获取帧尺寸 (width, height)"""
        if self.cap is not None:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return self.width, self.height

    def get_total_frames(self) -> int:
        """获取总帧数（仅视频文件有效，否则返回 -1）"""
        if self.cap is None or not self.is_file:
            return -1
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else -1

    def __bool__(self) -> bool:
        """
        是否已创建底层捕获对象（供 ``if stream:`` 判断）。

        实时源与文件源均可用；不会调用 :meth:`__len__`，避免 live 源在布尔上下文中抛错。
        """
        return self.cap is not None

    def __len__(self) -> int:
        """
        视频文件总帧数。

        Raises:
            RuntimeError: 流未打开
            TypeError: 实时源（摄像头/RTSP）无固定长度
        """
        if self.cap is None:
            raise RuntimeError("VideoStream is not open; call open() first")
        if not self.is_file:
            raise TypeError("len() is not defined for live video sources")
        total = self.get_total_frames()
        return max(0, total)

    def release(self) -> None:
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoStream":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
