"""工具函数包"""
from .config import load_config, get_config, ConfigValidationError
from .config_schema import validate_config
from .bbox import clamp_bbox
from .model_paths import resolve_yolo_model, resolve_ocr_model, PROJECT_ROOT
from .logging_config import setup_logging
from .performance import PerformanceOptimizer, FPSMonitor, DegradationPlan
from .model_manager import ModelManager
__all__ = [
    'load_config', 'get_config', 'validate_config', 'ConfigValidationError',
    'clamp_bbox',
    'resolve_yolo_model', 'resolve_ocr_model', 'PROJECT_ROOT',
    'setup_logging',
    'PerformanceOptimizer', 'FPSMonitor', 'DegradationPlan',
    'ModelManager',
]
