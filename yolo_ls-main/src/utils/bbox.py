"""边界框工具：裁剪、合法性检查。"""
from typing import Tuple, Optional
import numpy as np


def clamp_bbox(
    bbox: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    min_size: int = 1,
) -> Optional[Tuple[int, int, int, int]]:
    """
    将边界框裁剪到图像范围内；无效框返回 None。

    Args:
        bbox: (x1, y1, x2, y2)
        frame_w, frame_h: 图像宽高
        min_size: 最小宽/高（像素）
    """
    if frame_w <= 0 or frame_h <= 0:
        return None

    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(x1, frame_w - 1))
    y1 = max(0, min(y1, frame_h - 1))
    x2 = max(0, min(x2, frame_w))
    y2 = max(0, min(y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None

    return x1, y1, x2, y2


def clamp_bbox_array(
    xyxy: np.ndarray,
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    """从浮点数组裁剪为整数 bbox。"""
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy[:4])
    clamped = clamp_bbox((x1, y1, x2, y2), frame_w, frame_h)
    if clamped is None:
        return 0, 0, 0, 0
    return clamped
