"""配置管理模块"""
import yaml
from pathlib import Path
from typing import Any, Dict

_config: Dict[str, Any] = {}

def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    global _config
    path = Path(config_path)
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
    return _config

def get_config(key: str = None) -> Any:
    """获取配置项"""
    if not _config:
        load_config()
    if key is None:
        return _config
    keys = key.split('.')
    value = _config
    for k in keys:
        value = value.get(k, {})
    return value
