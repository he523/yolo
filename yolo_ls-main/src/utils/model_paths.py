"""模型路径解析：自定义权重缺失时回退到预训练。"""
import logging
from pathlib import Path
from typing import Optional, Tuple

from .constants import YOLO_FALLBACK_MODELS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str, base: Optional[Path] = None) -> Path:
    """相对项目根目录解析路径。"""
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    base = base or PROJECT_ROOT
    candidate = base / path
    if candidate.exists():
        return candidate
    return p


def resolve_yolo_model(model_path: str, base: Optional[Path] = None) -> Tuple[str, bool]:
    """
    解析 YOLO 权重路径。

    Returns:
        (实际使用的路径, 是否为回退模型)
    """
    base = base or PROJECT_ROOT
    primary = resolve_path(model_path, base)
    if primary.exists():
        logger.info("Using YOLO model: %s", primary)
        return str(primary), False

    logger.warning("YOLO model not found at %s, trying fallbacks", model_path)
    for name in YOLO_FALLBACK_MODELS:
        fb = base / name
        if fb.exists():
            logger.warning("Fallback to %s (train with scripts/train.py --output-model)", fb)
            return str(fb), True
        try:
            from ultralytics import YOLO
            YOLO(name)
            logger.warning("Downloaded/using Ultralytics default %s", name)
            return name, True
        except Exception as exc:
            logger.debug("Fallback %s failed: %s", name, exc)

    raise FileNotFoundError(
        f"No YOLO weights at {model_path}. Run: python scripts/download_models.py "
        f"or python scripts/train.py --output-model models/yolo12n_vehicle.pt"
    )


def resolve_ocr_model(model_path: str, base: Optional[Path] = None) -> Optional[Path]:
    """解析 OCR 权重；不存在则返回 None（由 PaddleOCR 兜底）。"""
    base = base or PROJECT_ROOT
    p = resolve_path(model_path, base)
    if p.exists():
        return p
    logger.warning("Plate OCR model not found at %s (PaddleOCR fallback if available)", model_path)
    return None
