"""统一日志配置。"""
import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """配置根 logger（仅首次有效）。"""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    ))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
