"""
红绿灯状态 CNN 分类器（推理模块）

替换原有的 HSV 颜色分析，使用 MobileNetV3-Small 模型对 YOLO 检出的
红绿灯 ROI 进行 4 分类：red / yellow / green / off。

若模型文件不存在或加载失败，自动回退到 HSV 模式。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# 与训练脚本保持一致
INPUT_SIZE = 96
CLASS_NAMES = ["green", "off", "red", "yellow"]  # 字母序，train 脚本中的顺序
# ROI 最小尺寸：任一维度小于此值或面积小于平方值时跳过 CNN，回退 HSV
MIN_ROI_DIM = 10
MIN_ROI_AREA = 400  # 20×20


class TrafficLightClassifier:
    """
    红绿灯状态 CNN 分类器。

    用法:
        clf = TrafficLightClassifier("models/traffic_light_cls.pt")
        state, conf = clf.classify(frame, bbox)
        # state: 'red' | 'yellow' | 'green' | 'off'
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: TorchScript (.pt) 模型文件路径
            device: 'cuda' / 'cpu'
        """
        self.model = None
        self.device = device
        self.model_path = model_path

        if not model_path:
            logger.info("TrafficLightClassifier: no model_path provided, will use HSV fallback")
            return

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """加载 TorchScript 模型"""
        import torch

        pt_path = Path(model_path)
        if not pt_path.exists():
            logger.warning(
                "TrafficLightClassifier model not found at %s — falling back to HSV",
                model_path,
            )
            self.model = None
            return

        try:
            device = self.device if torch.cuda.is_available() else "cpu"
            self.model = torch.jit.load(str(pt_path), map_location=device)
            self.model.eval()
            logger.info("TrafficLightClassifier loaded from %s on %s", model_path, device)
        except Exception as exc:
            logger.error("Failed to load TrafficLightClassifier: %s", exc)
            self.model = None

    @property
    def is_available(self) -> bool:
        return self.model is not None

    def classify(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, float]:
        """
        对红绿灯 ROI 进行分类。

        Args:
            frame: BGR 整帧图像
            bbox: (x1, y1, x2, y2) 红绿灯边界框

        Returns:
            (state: str, confidence: float)
            state ∈ {'red', 'yellow', 'green', 'off'}
        """
        if self.model is None:
            return "unknown", 0.0

        import torch
        from torchvision import transforms

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # ROI 最小尺寸检查：过小的 bbox resize 后极度模糊，CNN 无法可靠分类
        roi_w, roi_h = x2 - x1, y2 - y1
        if roi_w < MIN_ROI_DIM or roi_h < MIN_ROI_DIM or roi_w * roi_h < MIN_ROI_AREA:
            return "unknown", 0.0

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown", 0.0

        # 预处理：与训练脚本 EVAL_TRANSFORM 完全一致（PIL Resize → ToTensor → Normalize）
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        eval_transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = eval_transform(roi_pil).unsqueeze(0)
        tensor = tensor.to(next(self.model.parameters()).device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        state = CLASS_NAMES[pred.item()]
        confidence = float(conf.item())
        return state, confidence
