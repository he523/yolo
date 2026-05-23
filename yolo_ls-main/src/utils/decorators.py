"""通用装饰器。"""
import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def safe_operation(
    default_return: Any = None,
    log_level: str = 'warning',
    reraise: bool = False,
):
    """捕获异常、记录日志，可选返回默认值。"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log_func = getattr(logger, log_level, logger.warning)
                log_func("Error in %s: %s", func.__name__, exc, exc_info=log_level == 'debug')
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator
