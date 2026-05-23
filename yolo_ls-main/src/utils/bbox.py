"""边界框工具：裁剪、合法性检查、IOU 计算。"""
from typing import Tuple, Optional, Union
import numpy as np

_BBox = Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]


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


def iou_xyxy(a: _BBox, b: _BBox) -> float:
    """计算两个 xyxy 边界框的 IOU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def iou_xyxy_batch(a: _BBox, boxes: np.ndarray) -> np.ndarray:
    """
    计算框 a 与 boxes 中每一行的 IOU。

    Args:
        a: (x1, y1, x2, y2)
        boxes: shape (N, 4) 的 xyxy 数组

    Returns:
        shape (N,) 的 IOU 数组
    """
    if boxes.size == 0:
        return np.array([], dtype=np.float64)

    boxes = np.asarray(boxes, dtype=np.float64)
    ax1, ay1, ax2, ay2 = (float(v) for v in a)

    inter_x1 = np.maximum(ax1, boxes[:, 0])
    inter_y1 = np.maximum(ay1, boxes[:, 1])
    inter_x2 = np.minimum(ax2, boxes[:, 2])
    inter_y2 = np.minimum(ay2, boxes[:, 3])

    iw = np.maximum(0.0, inter_x2 - inter_x1)
    ih = np.maximum(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_a + area_b - inter + 1e-6
    return inter / union
