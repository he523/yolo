"""集中管理阈值与默认参数，避免魔法数字散落各处。"""
from typing import Dict, Tuple

# 违规 / 场景
DEFAULT_EMERGENCY_DISTANCE_PX = 300

# 逆行：与期望流向相反的方向对
OPPOSITE_DIRECTIONS: Dict[str, str] = {
    'north': 'south',
    'south': 'north',
    'east': 'west',
    'west': 'east',
    'northeast': 'southwest',
    'southwest': 'northeast',
    'northwest': 'southeast',
    'southeast': 'northwest',
}

# 逆行 / 违规变道
DEFAULT_LANE_CHANGE_LATERAL_PX = 80
DEFAULT_LANE_CHANGE_MIN_SPEED_KMH = 15.0
DEFAULT_LANE_CHANGE_HISTORY_LEN = 8

# 颜色 / HSV
COLOR_RATIO_THRESHOLD = 0.08
TRAFFIC_LIGHT_COLOR_THRESHOLD = 0.06
TRAFFIC_LIGHT_MIN_ASPECT = 0.25
TRAFFIC_LIGHT_MAX_ASPECT = 0.55

# 检测切片
DEFAULT_TILING_MIN_DETS = 10
DEFAULT_TILING_OVERLAP = 0.20
DEFAULT_TILING_INTERVAL_FRAMES = 5

# 数据库清理
DEFAULT_DB_RETENTION_DAYS = 30
DEFAULT_DB_MAX_ROWS_PER_TABLE = 50_000
DEFAULT_DB_MAX_SIZE_MB = 512

# 预训练回退（Ultralytics 官方权重名）
YOLO_FALLBACK_MODELS: Tuple[str, ...] = (
    'yolo12n.pt',
    'yolo11n.pt',
    'yolov8n.pt',
)
