"""HSV 颜色分析辅助：光照归一化。"""
import cv2
import numpy as np


def normalize_roi_for_hsv(roi: np.ndarray) -> np.ndarray:
    """
    对 ROI 做 CLAHE 亮度均衡，减轻光照变化对 HSV 阈值的影响。
    返回 BGR 图像（与输入通道一致）。
    """
    if roi is None or roi.size == 0:
        return roi

    if len(roi.shape) == 2:
        gray = roi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eq = clahe.apply(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def bgr_to_hsv_normalized(roi: np.ndarray) -> np.ndarray:
    """CLAHE 后再转 HSV。"""
    normalized = normalize_roi_for_hsv(roi)
    return cv2.cvtColor(normalized, cv2.COLOR_BGR2HSV)
