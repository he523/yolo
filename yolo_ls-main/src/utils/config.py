"""配置管理模块"""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.config_schema import ConfigValidationError, validate_config

logger = logging.getLogger(__name__)

__all__ = ["load_config", "get_config", "validate_config", "ConfigValidationError"]

_config: Dict[str, Any] = {}


def load_config(
    config_path: str = "config/settings.yaml",
    *,
    validate: bool = True,
    strict_unknown: bool = False,
) -> Dict[str, Any]:
    """
    加载并可选校验配置文件。

    Args:
        config_path: YAML 路径
        validate: 是否执行 schema 校验与默认值合并
        strict_unknown: 校验时是否拒绝未知顶层键

    Returns:
        配置字典

    Raises:
        ConfigValidationError: validate=True 且配置非法时
    """
    global _config
    path = Path(config_path)
    raw: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw = loaded
            elif loaded is not None:
                raise ConfigValidationError(
                    f"Config file root must be a mapping, got {type(loaded).__name__}"
                )
    if validate:
        try:
            _config = validate_config(raw, strict_unknown=strict_unknown)
        except ConfigValidationError:
            logger.exception("Invalid configuration in %s", path)
            raise
    else:
        _config = raw
    return _config


def get_config(key: Optional[str] = None) -> Any:
    """获取配置项；支持点分路径，如 ``video.fps``。"""
    if not _config:
        load_config()
    if key is None:
        return _config
    value: Any = _config
    for part in key.split("."):
        if not isinstance(value, dict):
            return {}
        value = value.get(part, {})
    return value
