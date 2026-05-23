"""Matplotlib 中文字体配置（避免 DejaVu Sans 缺字警告）。"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# 按平台常见字体优先级
_CJK_FONT_CANDIDATES: List[str] = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "PingFang SC",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
]

_configured = False
_chosen_font: Optional[str] = None


def _register_windows_system_fonts() -> Optional[str]:
    """从 Windows\\Fonts 注册常见中文字体文件。"""
    if sys.platform != "win32":
        return None
    from matplotlib import font_manager

    windir = os.environ.get("WINDIR", r"C:\Windows")
    font_dir = Path(windir) / "Fonts"
    for filename in ("msyh.ttc", "msyhbd.ttc", "msyhl.ttc", "simhei.ttf", "simsun.ttc"):
        path = font_dir / filename
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            if name:
                return name
        except (OSError, ValueError) as exc:
            logger.debug("Skip font %s: %s", path, exc)
    return None


def _resolve_cjk_font() -> Optional[str]:
    from matplotlib import font_manager

    registered = _register_windows_system_fonts()
    if registered:
        return registered

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONT_CANDIDATES:
        if name in available:
            return name

    lowered = {n.lower(): n for n in available}
    for name in _CJK_FONT_CANDIDATES:
        key = name.lower()
        for low, original in lowered.items():
            if key in low or low in key:
                return original
    return None


def configure_matplotlib_chinese(force: bool = False) -> str:
    """
    设置 matplotlib 使用系统中文字体。

    Returns:
        实际选用的字体名；若无可用中文字体则回退 DejaVu Sans。
    """
    global _configured, _chosen_font
    if _configured and not force:
        return _chosen_font or "DejaVu Sans"

    import matplotlib

    chosen = _resolve_cjk_font()
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans", "sans-serif"]
        matplotlib.rcParams["font.family"] = "sans-serif"
        logger.debug("Matplotlib CJK font: %s", chosen)
    else:
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
        logger.warning(
            "No CJK font found for matplotlib; chart Chinese labels may not render correctly"
        )
        chosen = "DejaVu Sans"

    matplotlib.rcParams["axes.unicode_minus"] = False
    _configured = True
    _chosen_font = chosen
    return chosen


def chart_font_props(**kwargs):
    """返回带中文字体的 text 属性 dict（用于 pie textprops 等）。"""
    configure_matplotlib_chinese()
    props = {"fontsize": kwargs.pop("fontsize", 8)}
    props.update(kwargs)
    if _chosen_font and _chosen_font != "DejaVu Sans":
        props["fontfamily"] = _chosen_font
    return props
