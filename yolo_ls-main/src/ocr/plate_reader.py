"""车牌 OCR 识别模块"""
import contextlib
import cv2
import logging
import numpy as np
import re
import os
import subprocess
import sys
import warnings
from typing import Tuple, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms

from src.utils.bbox import clamp_bbox
from src.utils.hsv import bgr_to_hsv_normalized
from src.utils.model_paths import resolve_ocr_model

logger = logging.getLogger(__name__)


def _patch_paddle_ccache_lookup() -> None:
    """避免 Paddle 在 Windows 上调用 where ccache 产生中文控制台噪声。"""
    try:
        import paddle.utils.cpp_extension.extension_utils as ext  # type: ignore
        ext.find_ccache_home = lambda: None  # type: ignore[method-assign]
    except Exception:
        pass


@contextlib.contextmanager
def _quiet_paddle_bootstrap():
    """屏蔽 Paddle 导入期的 ccache 查找与相关警告。"""
    orig_check = subprocess.check_output

    def _wrapped(cmd, *args, **kwargs):
        kwargs.setdefault('stderr', subprocess.DEVNULL)
        return orig_check(cmd, *args, **kwargs)

    subprocess.check_output = _wrapped
    try:
        yield
    finally:
        subprocess.check_output = orig_check


# 中国车牌字符集
CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '警', '学', '挂',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-'  # blank for CTC
]
IDX2CHAR = {idx: char for idx, char in enumerate(CHARS)}
BLANK_IDX = len(CHARS) - 1
PROVINCE_CHARS = set(CHARS[:34])

# Paddle 在 Windows + PIR/oneDNN 下会触发推理失败，必须关闭
_PADDLE_RUNTIME_KWARGS = {
    'enable_mkldnn': False,
    'enable_cinn': False,
    'device': 'cpu',
}
_PADDLE_PREP_PATCHED = False


def _configure_paddle_runtime() -> None:
    """
    在导入 paddleocr 之前配置运行环境。
    Windows 上 PIR + oneDNN 会报 ConvertPirAttribute2RuntimeAttribute 错误。
    """
    global _PADDLE_PREP_PATCHED
    if sys.platform == 'win32':
        for key, value in (
            ('FLAGS_enable_pir_api', '0'),
            ('FLAGS_enable_pir_in_executor', '0'),
            ('FLAGS_use_mkldnn', '0'),
            ('FLAGS_json_format_model', '0'),
            ('PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT', '0'),
        ):
            os.environ[key] = value

    if _PADDLE_PREP_PATCHED:
        return

    try:
        import paddleocr._common_args as paddle_common  # type: ignore
        from paddlex.inference.utils.pp_option import PaddlePredictorOption  # type: ignore

        if not getattr(PaddlePredictorOption.device_type, '_yolo_ls_patched', False):

            def _device_setter(self, device_type):  # noqa: ANN001
                if device_type not in PaddlePredictorOption.SUPPORT_DEVICE:
                    supported = ', '.join(PaddlePredictorOption.SUPPORT_DEVICE)
                    raise ValueError(
                        f"The device type must be one of {supported}, "
                        f"but received {repr(device_type)}."
                    )
                self._update('device_type', device_type)
                from paddlex.utils.device import set_env_for_device_type  # type: ignore
                set_env_for_device_type(device_type)
                if sys.platform != 'win32' and device_type in ('gpu', 'cpu'):
                    os.environ['FLAGS_enable_pir_api'] = '1'

            PaddlePredictorOption.device_type = property(
                PaddlePredictorOption.device_type.fget,
                _device_setter,
            )
            PaddlePredictorOption.device_type._yolo_ls_patched = True  # type: ignore[attr-defined]

        _orig_prepare = paddle_common.prepare_common_init_args

        def _prepare_common_init_args(model_name, common_args):  # noqa: ANN001
            args = dict(common_args)
            args['enable_mkldnn'] = False
            args['enable_cinn'] = False
            args['device'] = 'cpu'
            init_kwargs = _orig_prepare(model_name, args)
            pp_option = init_kwargs.get('pp_option')
            if pp_option is not None:
                pp_option.run_mode = 'paddle'
                pp_option.enable_new_ir = False
                pp_option.device_type = 'cpu'
            return init_kwargs

        paddle_common.prepare_common_init_args = _prepare_common_init_args
        _PADDLE_PREP_PATCHED = True
    except Exception as exc:
        logger.debug('Paddle runtime patch skipped: %s', exc)


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """增强小尺寸车牌 ROI，提高 OCR 召回。"""
    if image is None or image.size == 0:
        return image
    h, w = image.shape[:2]
    min_side = min(h, w)
    if min_side < 64:
        scale = 64.0 / min_side
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
        )
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    except cv2.error:
        return image


class CRNN(nn.Module):
    """CRNN 车牌识别模型"""

    def __init__(self, num_classes: int = len(CHARS), hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, (2, 1), 1, 0), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return output.permute(1, 0, 2)


@dataclass
class PlateResult:
    """车牌识别结果"""
    plate_number: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # 车牌位置


class PlateDetector:
    """车牌检测器（基于颜色和形态学）"""

    def __init__(self):
        # 蓝色车牌 HSV 范围
        self.blue_lower = np.array([100, 80, 80])
        self.blue_upper = np.array([130, 255, 255])
        # 黄色车牌 HSV 范围
        self.yellow_lower = np.array([15, 80, 80])
        self.yellow_upper = np.array([40, 255, 255])
        # 绿色车牌 HSV 范围（新能源）
        self.green_lower = np.array([35, 80, 80])
        self.green_upper = np.array([85, 255, 255])

    def detect(self, frame: np.ndarray,
               vehicle_bbox: Tuple[int, int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        在车辆区域内检测车牌

        Args:
            frame: BGR 图像
            vehicle_bbox: 车辆边界框

        Returns:
            车牌边界框或 None
        """
        fh, fw = frame.shape[:2]
        clamped = clamp_bbox(vehicle_bbox, fw, fh)
        if clamped is None:
            return None
        x1, y1, x2, y2 = clamped
        roi_y1 = y1 + (y2 - y1) // 2
        roi = frame[roi_y1:y2, x1:x2]

        if roi.size == 0:
            return None

        hsv = bgr_to_hsv_normalized(roi)

        # 检测蓝色、黄色、绿色车牌
        masks = [
            cv2.inRange(hsv, self.blue_lower, self.blue_upper),
            cv2.inRange(hsv, self.yellow_lower, self.yellow_upper),
            cv2.inRange(hsv, self.green_lower, self.green_upper),
        ]
        mask = masks[0] | masks[1] | masks[2]

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        best_score = 0

        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            rx, ry, rw, rh = rect

            # 车牌宽高比约 2:1~6:1，尺寸下限放宽以适配远处小车牌
            aspect_ratio = rw / rh if rh > 0 else 0
            if 2.0 < aspect_ratio < 6.0 and rw > 40 and rh > 10:
                area = rw * rh
                if area > best_score:
                    best_score = area
                    best_rect = (x1 + rx, roi_y1 + ry, x1 + rx + rw, roi_y1 + ry + rh)

        return best_rect


class PlateOCR:
    """车牌 OCR 识别器（使用 CRNN 模型）"""

    def __init__(
        self,
        model_path: str = "models/plate_ocr.pt",
        use_gpu: bool = True,
        paddle_mobile: bool = True,
    ):
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = None
        self._paddle_ocr = None
        self.paddle_mobile = bool(paddle_mobile)
        # PaddleOCR 固定 CPU，避免 Windows 上 GPU+PIR 触发 oneDNN 异常
        self._paddle_device = 'cpu'
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """加载 CRNN 模型"""
        path = resolve_ocr_model(model_path)
        if path is not None:
            self.model = CRNN().to(self.device)
            # torch<2.0 可能不支持 weights_only 参数
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info("Loaded plate OCR model from %s", path)
        else:
            logger.warning("Plate OCR model not found at %s (PaddleOCR fallback)", model_path)

    def _get_paddle_ocr(self):
        """懒加载 PaddleOCR（用于无 CRNN 权重时的兜底）"""
        if self._paddle_ocr is not None:
            return self._paddle_ocr
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        _configure_paddle_runtime()
        try:
            with _quiet_paddle_bootstrap():
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    from paddleocr import PaddleOCR  # type: ignore
                    _patch_paddle_ccache_lookup()
        except Exception as exc:
            logger.debug("PaddleOCR import failed: %s", exc)
            self._paddle_ocr = None
            return None

        self._paddle_ocr = self._create_paddle_ocr_instance(
            PaddleOCR, self.paddle_mobile, self._paddle_device,
        )
        if self._paddle_ocr is None:
            logger.warning("PaddleOCR init failed (see debug log for attempts)")
        else:
            logger.info("PaddleOCR initialized for plate fallback")
        return self._paddle_ocr

    @staticmethod
    def _create_paddle_ocr_instance(
        paddle_ocr_cls,
        paddle_mobile: bool = True,
        device: str = 'cpu',
    ) -> Optional[Any]:
        """
        兼容 PaddleOCR 2.x 与 3.x 构造参数。
        3.x 已移除 show_log，use_angle_cls 改为 use_textline_orientation。
        """
        doc_off = {
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
        }
        runtime = dict(_PADDLE_RUNTIME_KWARGS)
        if paddle_mobile:
            attempts = [
                {
                    'lang': 'ch',
                    'ocr_version': 'PP-OCRv4',
                    **doc_off,
                    **runtime,
                },
                {
                    'lang': 'ch',
                    'text_detection_model_name': 'PP-OCRv4_mobile_det',
                    'text_recognition_model_name': 'PP-OCRv4_mobile_rec',
                    **doc_off,
                    **runtime,
                },
                {'lang': 'ch', **doc_off, **runtime},
            ]
        else:
            attempts = [
                {'lang': 'ch', **doc_off, **runtime},
                {'lang': 'ch', **runtime},
                {'use_angle_cls': False, 'lang': 'ch', **runtime},
            ]
        for kwargs in attempts:
            try:
                with _quiet_paddle_bootstrap():
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UserWarning)
                        return paddle_ocr_cls(**kwargs)
            except TypeError as exc:
                logger.debug("PaddleOCR init skip %s: %s", kwargs, exc)
            except Exception as exc:
                logger.debug("PaddleOCR init failed %s: %s", kwargs, exc)
        return None

    @staticmethod
    def _run_paddle_inference(ocr: Any, image: np.ndarray) -> Optional[Any]:
        """兼容 2.x ocr(cls=) 与 3.x predict(use_textline_orientation=)。"""
        image = _preprocess_for_ocr(image)
        call_variants = [
            lambda: ocr.predict(image, use_textline_orientation=False),
            lambda: ocr.predict(image),
            lambda: ocr.ocr(image, use_textline_orientation=False),
            lambda: ocr.ocr(image, cls=False),
            lambda: ocr.ocr(image),
        ]
        last_err: Optional[Exception] = None
        for call in call_variants:
            try:
                result = call()
                if result is not None:
                    return result
            except TypeError as exc:
                last_err = exc
                continue
            except Exception as exc:
                last_err = exc
                logger.debug("PaddleOCR inference attempt failed: %s", exc)
        if last_err is not None:
            logger.warning("PaddleOCR inference failed: %s", last_err)
        return None

    def _decode(self, preds: torch.Tensor) -> Tuple[str, float]:
        """CTC 解码"""
        preds_softmax = torch.softmax(preds, dim=2)
        preds_max, preds_idx = preds_softmax.max(2)
        preds_idx = preds_idx.permute(1, 0).cpu().numpy()[0]
        preds_max = preds_max.permute(1, 0).cpu().numpy()[0]

        chars, confs = [], []
        prev = -1
        for i, p in enumerate(preds_idx):
            if p != prev and p != BLANK_IDX:
                chars.append(IDX2CHAR.get(p, ''))
                confs.append(preds_max[i])
            prev = p

        text = ''.join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        return text, conf

    def recognize(self, frame: np.ndarray,
                  plate_bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        fh, fw = frame.shape[:2]
        clamped = clamp_bbox(plate_bbox, fw, fh)
        if clamped is None:
            return None
        x1, y1, x2, y2 = clamped
        plate_img = frame[y1:y2, x1:x2]
        if plate_img.size == 0:
            return None

        # 1) 优先使用 CRNN（若权重存在）
        if self.model is not None:
            # BGR to RGB
            plate_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            try:
                img_tensor = self.transform(plate_rgb).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    preds = self.model(img_tensor)
                text, conf = self._decode(preds)

                # 对于界面展示，适当放宽过滤条件，避免长期无结果
                plate_number = self._clean_plate(text, allow_loose=True)
                if plate_number:
                    return PlateResult(plate_number=plate_number, confidence=conf, bbox=plate_bbox)
            except Exception as exc:
                logger.debug("CRNN recognize failed: %s", exc)

        # 2) PaddleOCR 兜底（无权重或CRNN失败）
        ocr = self._get_paddle_ocr()
        if ocr is None:
            return None

        result = self._run_paddle_inference(ocr, plate_img)
        if result is None:
            return None

        text, conf = self._extract_paddle_text(result)
        plate_number = self._clean_plate(text, allow_loose=True)
        if not plate_number:
            return None
        return PlateResult(plate_number=plate_number, confidence=conf, bbox=plate_bbox)

    def recognize_from_vehicle_roi(self, frame: np.ndarray,
                                   vehicle_bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        """
        当车牌定位失败时的兜底：直接对车辆下半区域做 PaddleOCR。
        """
        ocr = self._get_paddle_ocr()
        if ocr is None:
            return None

        x1, y1, x2, y2 = vehicle_bbox
        h = y2 - y1
        if h <= 0:
            return None
        roi_y1 = y1 + int(h * 0.45)
        roi = frame[roi_y1:y2, x1:x2]
        if roi.size == 0:
            return None

        result = self._run_paddle_inference(ocr, roi)
        if result is None:
            return None

        text, conf = self._extract_paddle_text(result)
        plate_number = self._clean_plate(text, allow_loose=True)
        if not plate_number:
            return None

        # 无法得到精确车牌框时，返回车辆ROI作为近似bbox（用于GUI显示即可）
        return PlateResult(plate_number=plate_number, confidence=conf, bbox=(x1, roi_y1, x2, y2))

    def _extract_paddle_text(self, result: Any) -> Tuple[str, float]:
        """
        从 PaddleOCR 输出提取文本与置信度。
        支持 2.x: [[[box], (text, score)], ...]
        支持 3.x: [OCRResult] 含 rec_texts / rec_scores
        """
        if not result:
            return "", 0.0

        texts: List[str] = []
        scores: List[float] = []

        pages = result if isinstance(result, list) else [result]
        for page in pages:
            rec_texts, rec_scores = self._parse_paddle_page(page)
            texts.extend(rec_texts)
            scores.extend(rec_scores)

        if not texts:
            return "", 0.0
        return "".join(texts), float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _parse_paddle_page(page: Any) -> Tuple[List[str], List[float]]:
        """解析单页/单图 OCR 结果。"""
        texts: List[str] = []
        scores: List[float] = []

        rec_texts = None
        rec_scores = None

        payload = page
        if isinstance(page, dict):
            payload = page.get('res', page)
        elif hasattr(page, 'json'):
            raw_json = page.json
            if isinstance(raw_json, dict):
                payload = raw_json.get('res', raw_json)

        if isinstance(payload, dict):
            rec_texts = payload.get('rec_texts')
            rec_scores = payload.get('rec_scores')
        else:
            for key in ('rec_texts', 'rec_scores'):
                try:
                    if key == 'rec_texts':
                        rec_texts = page['rec_texts']
                    else:
                        rec_scores = page['rec_scores']
                except (KeyError, TypeError):
                    pass

        if rec_texts is not None:
            for i, txt in enumerate(rec_texts):
                if not txt:
                    continue
                if isinstance(txt, tuple):
                    txt = txt[0]
                texts.append(str(txt))
                if rec_scores is not None and i < len(rec_scores):
                    try:
                        scores.append(float(rec_scores[i]))
                    except (TypeError, ValueError):
                        scores.append(0.0)
            return texts, scores

        # PaddleOCR 2.x: list of [box, (text, score)]
        items = page
        if isinstance(page, list) and page and isinstance(page[0], list):
            if page and len(page) > 0 and not isinstance(page[0], (list, tuple)):
                items = page
            elif page and isinstance(page[0], list) and page[0] and len(page[0]) == 2:
                items = page[0] if isinstance(page[0][0], (list, tuple, np.ndarray)) else page

        if not isinstance(items, list):
            return texts, scores

        for it in items:
            try:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    text_part = it[1]
                    if isinstance(text_part, (list, tuple)) and len(text_part) >= 2:
                        txt, score = text_part[0], text_part[1]
                    else:
                        continue
                    if txt:
                        texts.append(str(txt))
                        scores.append(float(score))
            except (IndexError, TypeError, ValueError):
                continue
        return texts, scores

    def _clean_plate(self, text: str, allow_loose: bool = False) -> Optional[str]:
        """
        清理和验证车牌号

        Args:
            text: OCR 识别的原始文本

        Returns:
            清理后的车牌号或 None
        """
        # 保留汉字、字母、数字，移除其他字符
        text = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text.upper())
        text = text.replace('O', '0').replace('I', '1').replace('L', '1')

        if len(text) < 5:
            return None

        # 标准中国大陆车牌：省份简称 + 字母 + 5~6 位字母数字
        pattern = r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}$'
        if re.match(pattern, text):
            return text

        if allow_loose:
            # 放宽匹配：允许缺省份、长度 4~6 位字母数字、或含汉字超长文本
            if re.match(r'^[\u4e00-\u9fa5][A-Z][A-Z0-9]{4,6}$', text):
                return text
            if re.match(r'^[A-Z][A-Z0-9]{4,6}$', text):
                return text
            # 至少包含字母和数字的任意 ≥5 字符文本（PaddleOCR 可能输出不完整）
            if 5 <= len(text) <= 10:
                has_letter = bool(re.search(r'[A-Z]', text))
                has_digit = bool(re.search(r'[0-9]', text))
                if has_letter or has_digit:
                    return text

        return None


class PlateReader:
    """车牌识别器（整合检测和 OCR）"""

    def __init__(
        self,
        model_path: str = "models/plate_ocr.pt",
        use_gpu: bool = True,
        paddle_mobile: bool = True,
    ):
        self.detector = PlateDetector()
        self.ocr = PlateOCR(model_path, use_gpu, paddle_mobile=paddle_mobile)

    def read(self, frame: np.ndarray,
             vehicle_bbox: Tuple[int, int, int, int]) -> Optional[PlateResult]:
        """
        读取车牌

        Args:
            frame: BGR 图像
            vehicle_bbox: 车辆边界框

        Returns:
            PlateResult 或 None
        """
        plate_bbox = self.detector.detect(frame, vehicle_bbox)
        if plate_bbox is not None:
            result = self.ocr.recognize(frame, plate_bbox)
            if result is not None:
                return result
        return self.ocr.recognize_from_vehicle_roi(frame, vehicle_bbox)


# 模块加载时预先设置 Windows Paddle 环境（须在 paddleocr 首次 import 前）
_configure_paddle_runtime()
