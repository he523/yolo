"""GUI 国际化：简体中文 / English。"""
from __future__ import annotations

from typing import Any, Dict, Optional

SUPPORTED_LANGUAGES = ("zh_CN", "en")

_MESSAGES: Dict[str, Dict[str, str]] = {
    "zh_CN": {
        "app_title": "实时交通分析系统",
        "app_subtitle": "YOLO 检测 · ByteTrack 跟踪 · 自适应违规 · 碰撞风险预警",
        "live_idle": "● 待机",
        "live_running": "● 分析中",
        "language": "语言",
        "lang_zh": "简体中文",
        "lang_en": "English",
        "video_placeholder": "实时画面\n\n点击下方「打开视频」或「打开摄像头」开始分析\n支持 MP4 / AVI / MOV / MKV",
        "btn_open_video": "打开视频",
        "btn_open_camera": "打开摄像头",
        "btn_stop": "停止",
        "btn_pause": "暂停",
        "btn_resume": "继续",
        "tab_realtime": "实时",
        "tab_violations": "违规",
        "tab_database": "数据库",
        "tab_exemption": "免责说明",
        "tab_statistics": "统计",
        "tab_settings": "设置",
        "group_realtime": "实时概览",
        "group_vehicles": "检测车辆",
        "card_vehicles": "在途车辆",
        "card_speed": "平均速度",
        "card_emergency": "特种车辆",
        "card_fps": "运行帧率",
        "perf_line": "推理 · imgsz {imgsz} · 降级 L{level} · 跳帧 ×{skip}",
        "violations_title": "违规统计",
        "total_violations": "累计违规：{n}",
        "actual_violations": "实际违规：{n}",
        "exempted": "免责（特殊情形）：{n}",
        "risk_title": "碰撞风险",
        "risk_level": "风险等级：{level}",
        "risk_safe": "安全",
        "risk_low": "低",
        "risk_medium": "中",
        "risk_high": "高",
        "risk_critical": "严重",
        "min_ttc": "最短 TTC：{s}",
        "min_ttc_none": "最短 TTC：--",
        "active_risks": "活跃风险对：{n}",
        "active_risks_detail": "活跃风险对：{total}（严重 {critical} · 高 {high}）",
        "show_exempted": "显示免责记录",
        "only_exempted": "仅看免责",
        "search_plate_ph": "输入车牌号检索…",
        "btn_search": "搜索",
        "db_table_label": "数据表:",
        "db_search_ph": "按车牌筛选（仅 vehicles/violations 生效）",
        "db_refresh": "刷新",
        "db_delete": "删除选中车辆",
        "db_clean_label": "清空所选数据表:",
        "db_clean_btn": "清空所选表",
        "btn_reset_stats": "重置图表",
        "group_detection": "检测参数",
        "label_confidence": "置信度:",
        "label_speed_limit": "限速:",
        "label_emergency_dist": "特种车距离:",
        "label_flow_dir": "合法车流方向:",
        "cb_tiling": "启用切片检测（更慢，远处小目标更准）",
        "cb_wrong_way": "启用逆行检测",
        "cb_illegal_lane": "启用违规变道检测",
        "cb_performance": "启用自适应性能（FPS 监控与降级）",
        "btn_reload_cfg": "重新加载 settings.yaml",
        "group_stopline": "停止线",
        "cb_stopline": "启用停止线（闯红灯）",
        "label_sl_y": "Y 坐标:",
        "label_sl_x": "X 范围:",
        "btn_auto_stopline": "自动识别停止线",
        "status_ready": "就绪 · 请选择视频源开始分析",
        "status_processing": "分析中 · 点击「暂停」暂停",
        "status_paused": "已暂停 · 点击「继续」恢复",
        "status_stopped": "已停止",
        "status_settings_applied": "设置已实时生效",
        "status_settings_saved": "设置已保存，开始分析后生效",
        "status_stats_reset": "统计已重置",
        "dialog_select_video": "选择视频文件",
        "filter_video": "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        "error_open_source": "无法打开视频源",
        "chart_flow": "车流量",
        "chart_violation": "违规类型",
        "chart_speed": "速度分布",
        "chart_type": "车型分布",
        "chart_y_vehicles": "辆",
        "chart_no_data": "暂无数据",
        "veh_header_id": "ID",
        "veh_header_type": "车型",
        "veh_header_color": "颜色",
        "veh_header_speed": "速度",
        "veh_header_dir": "方向",
        "veh_header_plate": "车牌",
        "vio_header_time": "时间",
        "vio_header_type": "类型",
        "vio_header_plate": "车牌",
        "vio_header_speed": "速度",
        "vio_header_status": "状态",
        "vio_header_reason": "原因",
        "vio_header_details": "详情",
        "plate_recognizing": "识别中",
        "exemption_html": """
<h3 style="color:#ffffff;">自适应违规检测 · 免责情形</h3>
<p style="color:#a3a3a3;">系统识别以下特殊场景并标记为免责，仍保留快照供复核：</p>
<h4 style="color:#ffffff;">1. 避让特种车辆</h4>
<p style="color:#d4d4d4;">附近存在救护车、消防车、警车时，相关违规可记为免责。</p>
<h4 style="color:#ffffff;">2. 信号灯故障</h4>
<p style="color:#d4d4d4;">信号灯异常时，相关违规记为信号故障免责。</p>
<h4 style="color:#ffffff;">3. 其他情形</h4>
<ul style="color:#d4d4d4;"><li>交警指挥</li><li>紧急避险</li><li>施工绕行</li></ul>
""",
        "stopline_title": "自动识别停止线",
        "stopline_no_frame": "当前没有视频帧，请先打开视频或摄像头。",
        "stopline_bad_frame": "当前帧尺寸无效，无法检测停止线。",
        "stopline_not_found": "未检测到明显的水平停车线，请暂停画面后重试或手动设置。",
        "stopline_unstable": "未检测到稳定的停车线候选，请稍后再试或手动设置。",
        "stopline_ok": "已自动检测停止线：Y = {y}, X 范围 [{x1}, {x2}]。\n可在设置中微调后重新开始检测。",
        "db_error_title": "数据库错误",
        "db_delete_title": "删除车辆记录",
        "db_delete_only_vehicles": "当前仅在 vehicles 表中支持删除操作。",
        "db_delete_select": "请先在表格中选择要删除的记录。",
        "db_confirm_delete_title": "确认删除",
        "db_confirm_delete": "确定要删除选中的 {n} 条车辆记录吗？此操作不可恢复。",
        "db_deleted_title": "删除完成",
        "db_deleted": "已删除 {n} 条车辆记录。",
        "db_confirm_clean_title": "确认清理",
        "db_confirm_clean_table": "确定要清空 {table} 表的所有记录吗？此操作不可恢复。",
        "sec_unit": "秒",
        "status_review": "待复核",
        "status_violation": "违规",
        "search_found": "找到 {n} 条匹配记录",
        "search_not_found": "未找到匹配记录",
        "clean_done_table": "已清空 {table} 表，共删除 {count} 条记录。",
        "error_title": "错误",
        "clean_complete_title": "清理完成",
    },
    "en": {
        "app_title": "Real-Time Traffic Analysis",
        "app_subtitle": "YOLO · ByteTrack · Adaptive Violations · Collision Risk",
        "live_idle": "● Idle",
        "live_running": "● Running",
        "language": "Language",
        "lang_zh": "简体中文",
        "lang_en": "English",
        "video_placeholder": "Live View\n\nClick Open Video or Open Camera below to start\nSupports MP4 / AVI / MOV / MKV",
        "btn_open_video": "Open Video",
        "btn_open_camera": "Open Camera",
        "btn_stop": "Stop",
        "btn_pause": "Pause",
        "btn_resume": "Resume",
        "tab_realtime": "Live",
        "tab_violations": "Violations",
        "tab_database": "Database",
        "tab_exemption": "Exemptions",
        "tab_statistics": "Statistics",
        "tab_settings": "Settings",
        "group_realtime": "Overview",
        "group_vehicles": "Detected Vehicles",
        "card_vehicles": "Vehicles",
        "card_speed": "Avg Speed",
        "card_emergency": "Emergency",
        "card_fps": "FPS",
        "perf_line": "Infer · imgsz {imgsz} · L{level} · skip ×{skip}",
        "violations_title": "Violations",
        "total_violations": "Total: {n}",
        "actual_violations": "Actual: {n}",
        "exempted": "Exempted: {n}",
        "risk_title": "Collision Risk",
        "risk_level": "Risk: {level}",
        "risk_safe": "Safe",
        "risk_low": "Low",
        "risk_medium": "Medium",
        "risk_high": "High",
        "risk_critical": "Critical",
        "min_ttc": "Min TTC: {s}",
        "min_ttc_none": "Min TTC: --",
        "active_risks": "Active pairs: {n}",
        "active_risks_detail": "Active pairs: {total} (Critical {critical}, High {high})",
        "show_exempted": "Show exempted",
        "only_exempted": "Exempted only",
        "search_plate_ph": "Search by plate…",
        "btn_search": "Search",
        "db_table_label": "Table:",
        "db_search_ph": "Filter by plate (vehicles/violations)",
        "db_refresh": "Refresh",
        "db_delete": "Delete selected",
        "db_clean_label": "Clear selected table:",
        "db_clean_btn": "Clear table",
        "btn_reset_stats": "Reset charts",
        "group_detection": "Detection",
        "label_confidence": "Confidence:",
        "label_speed_limit": "Speed limit:",
        "label_emergency_dist": "Emergency distance:",
        "label_flow_dir": "Legal flow direction:",
        "cb_tiling": "Enable tiling (slower, better for small/distant objects)",
        "cb_wrong_way": "Enable wrong-way detection",
        "cb_illegal_lane": "Enable illegal lane-change detection",
        "cb_performance": "Enable adaptive performance (FPS degradation)",
        "btn_reload_cfg": "Reload settings.yaml",
        "group_stopline": "Stop line",
        "cb_stopline": "Enable stop line (red-light)",
        "label_sl_y": "Y position:",
        "label_sl_x": "X range:",
        "btn_auto_stopline": "Auto-detect stop line",
        "status_ready": "Ready · Select a video source to start",
        "status_processing": "Processing · Click Pause to pause",
        "status_paused": "Paused · Click Resume to continue",
        "status_stopped": "Stopped",
        "status_settings_applied": "Settings applied immediately",
        "status_settings_saved": "Settings saved; they apply when analysis starts",
        "status_stats_reset": "Statistics reset",
        "dialog_select_video": "Select video file",
        "filter_video": "Video (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        "error_open_source": "Cannot open video source",
        "chart_flow": "Traffic flow",
        "chart_violation": "Violation types",
        "chart_speed": "Speed distribution",
        "chart_type": "Vehicle types",
        "chart_y_vehicles": "count",
        "chart_no_data": "No data",
        "veh_header_id": "ID",
        "veh_header_type": "Type",
        "veh_header_color": "Color",
        "veh_header_speed": "Speed",
        "veh_header_dir": "Direction",
        "veh_header_plate": "Plate",
        "vio_header_time": "Time",
        "vio_header_type": "Type",
        "vio_header_plate": "Plate",
        "vio_header_speed": "Speed",
        "vio_header_status": "Status",
        "vio_header_reason": "Reason",
        "vio_header_details": "Details",
        "plate_recognizing": "Recognizing",
        "exemption_html": """
<h3 style="color:#ffffff;">Adaptive violations · Exemptions</h3>
<p style="color:#a3a3a3;">Special cases are marked exempt; snapshots are still saved.</p>
<h4 style="color:#ffffff;">1. Emergency vehicles</h4>
<p style="color:#d4d4d4;">Nearby ambulance, fire truck, or police may exempt related violations.</p>
<h4 style="color:#ffffff;">2. Signal malfunction</h4>
<p style="color:#d4d4d4;">Abnormal traffic lights may exempt related violations.</p>
<h4 style="color:#ffffff;">3. Other</h4>
<ul style="color:#d4d4d4;"><li>Police direction</li><li>Emergency avoidance</li><li>Construction detour</li></ul>
""",
        "stopline_title": "Auto-detect stop line",
        "stopline_no_frame": "No frame available. Open video or camera first.",
        "stopline_bad_frame": "Invalid frame size; cannot detect stop line.",
        "stopline_not_found": "No clear horizontal stop line found. Pause on the line or set manually.",
        "stopline_unstable": "No stable stop line candidate. Retry or set manually.",
        "stopline_ok": "Stop line detected: Y = {y}, X [{x1}, {x2}].\nFine-tune in Settings and restart.",
        "db_error_title": "Database error",
        "db_delete_title": "Delete vehicles",
        "db_delete_only_vehicles": "Delete is only supported on the vehicles table.",
        "db_delete_select": "Select rows in the table first.",
        "db_confirm_delete_title": "Confirm delete",
        "db_confirm_delete": "Delete {n} selected vehicle record(s)? This cannot be undone.",
        "db_deleted_title": "Delete complete",
        "db_deleted": "Deleted {n} vehicle record(s).",
        "db_confirm_clean_title": "Confirm cleanup",
        "db_confirm_clean_table": "Delete ALL records from table '{table}'? This cannot be undone.",
        "sec_unit": "s",
        "status_review": "Review",
        "status_violation": "Violation",
        "search_found": "Found {n} matching record(s)",
        "search_not_found": "No matching records found",
        "clean_done_table": "Cleared table '{table}': {count} record(s) deleted.",
        "error_title": "Error",
        "clean_complete_title": "Cleanup complete",
    },
}

_RISK_LEVEL_KEYS = {
    "safe": "risk_safe",
    "low": "risk_low",
    "medium": "risk_medium",
    "high": "risk_high",
    "critical": "risk_critical",
}


class Translator:
    """简单键值翻译器。"""

    def __init__(self, language: str = "zh_CN"):
        self.set_language(language)

    @property
    def language(self) -> str:
        return self._lang

    def set_language(self, language: str) -> None:
        if language not in SUPPORTED_LANGUAGES:
            language = "zh_CN"
        self._lang = language

    def tr(self, key: str, **kwargs: Any) -> str:
        text = _MESSAGES.get(self._lang, _MESSAGES["zh_CN"]).get(key, key)
        if kwargs:
            try:
                return text.format(**kwargs)
            except (KeyError, ValueError):
                return text
        return text

    def risk_level_label(self, level_value: str) -> str:
        key = _RISK_LEVEL_KEYS.get(level_value.lower(), "risk_safe")
        return self.tr(key)

    def plate_pending_token(self) -> str:
        return self.tr("plate_recognizing")

    def is_plate_pending(self, plate: Optional[str]) -> bool:
        if not plate:
            return False
        return plate in (
            _MESSAGES["zh_CN"]["plate_recognizing"],
            _MESSAGES["en"]["plate_recognizing"],
        )
