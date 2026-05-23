"""GUI 主题：黑底白字、全局 QSS。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QFont

if TYPE_CHECKING:
    from src.gui.i18n import Translator

_GUI_DIR = Path(__file__).resolve().parent
_CHECKBOX_CHECK_ICON = QUrl.fromLocalFile(
    str(_GUI_DIR / "assets" / "checkbox_check.png")
).toString()

# 黑底白字
COLORS = {
    "bg": "#000000",
    "bg_elevated": "#0a0a0a",
    "surface": "#111111",
    "surface_hover": "#1a1a1a",
    "border": "rgba(255, 255, 255, 0.14)",
    "border_focus": "rgba(255, 255, 255, 0.5)",
    "text": "#ffffff",
    "text_muted": "#a3a3a3",
    "text_dim": "#737373",
    "accent": "#ffffff",
    "accent_soft": "rgba(255, 255, 255, 0.1)",
    "success": "#22c55e",
    "success_soft": "rgba(34, 197, 94, 0.18)",
    "checkbox_border": "#737373",
    "chart_bg": "#000000",
    "chart_grid": "#333333",
}


def global_stylesheet() -> str:
    c = COLORS
    return f"""
        QMainWindow {{ background-color: {c['bg']}; }}
        QWidget {{
            font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
            font-size: 13px;
            color: {c['text']};
            background-color: transparent;
        }}
        QLabel {{ color: {c['text']}; background: transparent; }}

        QFrame#AppHeader {{
            background-color: {c['bg_elevated']};
            border: 1px solid {c['border']};
            border-radius: 8px;
        }}
        QLabel#HeaderTitle {{
            font-size: 18px;
            font-weight: 700;
            color: {c['text']};
        }}
        QLabel#HeaderSubtitle {{
            font-size: 11px;
            color: {c['text_muted']};
        }}
        QLabel#LiveBadge {{
            background-color: {c['surface']};
            color: {c['text_muted']};
            border: 1px solid {c['border']};
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 600;
        }}
        QLabel#LangLabel {{
            color: {c['text_muted']};
            font-size: 12px;
        }}

        QFrame#VideoPanel {{
            background-color: {c['bg_elevated']};
            border: 1px solid {c['border']};
            border-radius: 8px;
        }}
        QLabel#VideoPlaceholder {{
            color: {c['text_muted']};
            font-size: 13px;
        }}

        QGroupBox {{
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: 8px;
            margin-top: 10px;
            padding: 10px 12px 14px 12px;
            font-weight: 600;
            background-color: {c['surface']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: {c['text']};
        }}

        QTabWidget::pane {{
            border: 1px solid {c['border']};
            border-radius: 6px;
            background-color: {c['bg_elevated']};
        }}
        QTabBar::tab {{
            background: {c['surface']};
            color: {c['text_muted']};
            padding: 8px 16px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{
            background: {c['bg_elevated']};
            color: {c['text']};
            border-bottom: 2px solid {c['text']};
        }}
        QTabBar::tab:hover:!selected {{
            color: {c['text']};
            background: {c['surface_hover']};
        }}

        QSplitter::handle {{ background: {c['border']}; width: 2px; }}

        QTableWidget {{
            background-color: {c['bg_elevated']};
            alternate-background-color: {c['surface']};
            gridline-color: {c['border']};
            color: {c['text']};
            selection-background-color: {c['accent_soft']};
            selection-color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: 6px;
        }}
        QHeaderView::section {{
            background-color: {c['surface']};
            color: {c['text_muted']};
            padding: 8px;
            border: none;
            border-bottom: 1px solid {c['border']};
        }}

        QLineEdit, QSpinBox, QComboBox {{
            background-color: {c['bg_elevated']};
            border: 1px solid {c['border']};
            border-radius: 4px;
            padding: 6px 10px;
            color: {c['text']};
        }}
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border-color: {c['border_focus']};
        }}
        QComboBox QAbstractItemView {{
            background-color: {c['surface']};
            color: {c['text']};
            selection-background-color: {c['accent_soft']};
            border: 1px solid {c['border']};
        }}

        QCheckBox {{
            color: {c['text_muted']};
            spacing: 10px;
            padding: 3px 0;
        }}
        QCheckBox:checked {{
            color: {c['text']};
            font-weight: 600;
        }}
        QCheckBox:disabled {{
            color: {c['text_dim']};
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {c['checkbox_border']};
            border-radius: 4px;
            background: {c['bg_elevated']};
        }}
        QCheckBox::indicator:unchecked:hover {{
            border-color: {c['border_focus']};
            background: {c['surface_hover']};
        }}
        QCheckBox::indicator:checked {{
            background: {c['success']};
            border-color: {c['success']};
            image: url("{_CHECKBOX_CHECK_ICON}");
        }}
        QCheckBox::indicator:checked:hover {{
            background: #16a34a;
            border-color: #16a34a;
        }}
        QCheckBox::indicator:disabled {{
            border-color: {c['border']};
            background: {c['surface']};
        }}
        QCheckBox::indicator:checked:disabled {{
            background: #3f3f46;
            border-color: #3f3f46;
        }}

        QPushButton {{
            background-color: {c['surface']};
            color: {c['text']};
            border: 1px solid {c['border']};
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background-color: {c['surface_hover']};
            border-color: {c['border_focus']};
        }}
        QPushButton:pressed {{ background-color: {c['accent_soft']}; }}
        QPushButton:disabled {{
            color: {c['text_dim']};
            background-color: {c['surface']};
        }}
        QPushButton[btnRole="primary"] {{
            background-color: {c['text']};
            color: {c['bg']};
            border-color: {c['text']};
        }}
        QPushButton[btnRole="primary"]:hover {{
            background-color: #e5e5e5;
        }}
        QPushButton[btnRole="danger"] {{
            background-color: transparent;
            color: {c['text']};
            border-color: {c['text_muted']};
        }}

        QScrollBar:vertical {{
            background: transparent; width: 8px; margin: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: #404040; border-radius: 4px; min-height: 24px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

        QStatusBar {{
            background-color: {c['bg_elevated']};
            color: {c['text_muted']};
            border-top: 1px solid {c['border']};
        }}
    """


def insight_panel_style() -> str:
    return (
        f"background-color: {COLORS['surface']};"
        f"border-radius: 6px;"
        f"border: 1px solid {COLORS['border']};"
    )


def message_box_stylesheet() -> str:
    c = COLORS
    return f"""
        QMessageBox {{ background-color: {c['bg_elevated']}; }}
        QLabel {{ color: {c['text']}; }}
        QPushButton {{
            background-color: {c['text']};
            color: {c['bg']};
            border-radius: 4px;
            padding: 8px 18px;
            min-width: 72px;
        }}
        QPushButton:hover {{ background-color: #d4d4d4; }}
    """


class MetricCard(QFrame):
    """指标卡片。"""

    def __init__(
        self,
        title: str,
        value: str = "--",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setStyleSheet(f"""
            QFrame#MetricCard {{
                background-color: {COLORS['surface']};
                border-radius: 6px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        self.value_label = QLabel(value)
        self.value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        font = QFont("Microsoft YaHei UI", 17)
        font.setBold(True)
        self.value_label.setFont(font)
        self.value_label.setStyleSheet(f"color: {COLORS['text']};")

        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_title(self, title: str) -> None:
        self.title_label.setText(title)

    def set_value(self, text: str) -> None:
        self.value_label.setText(text)


def build_app_header(
    tr: "Translator",
) -> Tuple[QFrame, QLabel, QLabel, QLabel, QHBoxLayout]:
    """顶部栏：标题 + 右侧插槽（语言、状态）。"""
    header = QFrame()
    header.setObjectName("AppHeader")
    layout = QHBoxLayout(header)
    layout.setContentsMargins(16, 12, 16, 12)

    text_col = QVBoxLayout()
    text_col.setSpacing(2)
    title = QLabel(tr.tr("app_title"))
    title.setObjectName("HeaderTitle")
    subtitle = QLabel(tr.tr("app_subtitle"))
    subtitle.setObjectName("HeaderSubtitle")
    text_col.addWidget(title)
    text_col.addWidget(subtitle)

    right_box = QHBoxLayout()
    right_box.setSpacing(10)
    badge = QLabel(tr.tr("live_idle"))
    badge.setObjectName("LiveBadge")

    layout.addLayout(text_col, stretch=1)
    layout.addLayout(right_box)
    right_box.addWidget(badge)

    return header, title, subtitle, badge, right_box
