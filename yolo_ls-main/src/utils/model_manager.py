"""统一模型路径解析、校验与下载。"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .model_paths import PROJECT_ROOT, resolve_yolo_model, resolve_ocr_model, YOLO_FALLBACK_MODELS

logger = logging.getLogger(__name__)


class ModelManager:
    """
    集中管理项目模型文件。
    YOLO 缺失时回退 Ultralytics 预训练；plate_ocr 可选下载占位。
    """

    DEFAULT_PATHS: Dict[str, str] = {
        'yolo12n_vehicle': 'models/yolo12n_vehicle.pt',
        'plate_ocr': 'models/plate_ocr.pt',
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, root: Optional[Path] = None):
        self.root = root or PROJECT_ROOT
        self.config = config or {}
        self._cache: Dict[str, Any] = {}

    def get_path(self, model_name: str) -> Path:
        key = model_name.replace('.', '_')
        det_path = self.config.get('detector', {}).get('model_path')
        ocr_path = self.config.get('ocr', {}).get('model_path')
        if model_name in ('yolo12n_vehicle', 'vehicle', 'yolo'):
            return self.root / (det_path or self.DEFAULT_PATHS['yolo12n_vehicle'])
        if model_name in ('plate_ocr', 'ocr', 'plate'):
            return self.root / (ocr_path or self.DEFAULT_PATHS['plate_ocr'])
        return self.root / self.DEFAULT_PATHS.get(model_name, f'models/{model_name}.pt')

    def ensure_model(self, model_name: str) -> Path:
        path = self.get_path(model_name)
        if path.exists():
            return path
        logger.info("Model missing at %s, attempting download/fallback", path)
        self._download_model(model_name, path)
        return path

    def load_yolo_path(self, override: Optional[str] = None) -> Tuple[str, bool]:
        """返回 YOLO 可用路径及是否为回退权重。"""
        raw = override or str(self.get_path('yolo12n_vehicle'))
        resolved, fallback = resolve_yolo_model(raw, self.root)
        return resolved, fallback

    def load_ocr_path(self) -> Optional[Path]:
        return resolve_ocr_model(str(self.get_path('plate_ocr')), self.root)

    def _download_model(self, model_name: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if model_name in ('yolo12n_vehicle', 'vehicle', 'yolo'):
            return self._download_yolo_fallback(dest)
        if model_name in ('plate_ocr', 'ocr', 'plate'):
            logger.warning(
                "Plate OCR weights must be trained or placed at %s (no public URL configured)",
                dest,
            )
            return False
        return False

    def _download_yolo_fallback(self, dest: Path) -> bool:
        for name in YOLO_FALLBACK_MODELS:
            try:
                from ultralytics import YOLO
                logger.info("Downloading fallback YOLO: %s", name)
                model = YOLO(name)
                src = Path(getattr(model, 'ckpt_path', None) or self.root / name)
                if not src.exists():
                    src = self.root / name
                if src.exists():
                    shutil.copy2(src, dest)
                    logger.info("Saved YOLO model to %s", dest)
                    return True
            except Exception as exc:
                logger.debug("Fallback %s failed: %s", name, exc)
        return False
