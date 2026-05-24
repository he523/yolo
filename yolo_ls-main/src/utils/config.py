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


def save_config_section(
    config_path: str,
    section: str,
    key: str,
    value: Any,
) -> bool:
    """
    将嵌套配置项写回 YAML 文件，通过文本替换**完整保留注释和格式**。

    Args:
        config_path: YAML 文件路径
        section: 顶层键名，如 ``"violation"``
        key: 二级键名，如 ``"stop_line"``
        value: 要写入的值（dict/list/scalar）

    Returns:
        是否成功写入
    """
    import re as _re
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # 找出 key 在文件中的缩进深度（先于 value_block 生成）
        key_pattern = _re.compile(
            r"^([ \t]+)" + _re.escape(key) + r":",
            _re.MULTILINE,
        )
        km = key_pattern.search(text)
        if km is None:
            logger.warning("Key %s.%s not found in %s", section, key, path)
            return False
        base_indent = km.group(1)           # key 自身的缩进

        # 生成值的 YAML 表示，缩进与文件中 key 对齐
        value_lines = yaml.dump(
            {key: value},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).rstrip("\n")
        value_block = "\n".join(
            base_indent + line for line in value_lines.split("\n")
        )
        min_sub = len(base_indent) + 1      # 子行至少比 key 多 1 空格

        # 匹配 key: 行及其所有更深缩进的子行
        block_re = _re.compile(
            _re.escape(base_indent + key) + r":.*"
            r"(?:\n[ \t]{" + str(min_sub) + r",}.*)*",
        )

        new_text, count = _re.subn(block_re, value_block, text)
        if count == 0:
            logger.warning("Key %s.%s not found in %s", section, key, path)
            return False

        with open(path, "w", encoding="utf-8") as f:
            f.write(new_text)
        return True
    except Exception:
        logger.exception("Failed to save %s.%s to %s", section, key, path)
        return False


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
