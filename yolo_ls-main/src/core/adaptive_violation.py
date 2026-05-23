"""
自适应违规检测模块

创新点：智能识别特殊情况
- 检测特种车辆（救护车、消防车、警车）
- 检测交警指挥
- 如果违规时附近有特种车辆或交警，标记为"异常"待人工复核
- 异常情况单独保存截图，便于后续人工审核
"""
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

from src.utils.bbox import clamp_bbox
from src.utils.constants import (
    COLOR_RATIO_THRESHOLD,
    TRAFFIC_LIGHT_COLOR_THRESHOLD,
    TRAFFIC_LIGHT_MIN_ASPECT,
    TRAFFIC_LIGHT_MAX_ASPECT,
    DEFAULT_LANE_CHANGE_LATERAL_PX,
    DEFAULT_LANE_CHANGE_MIN_SPEED_KMH,
    DEFAULT_LANE_CHANGE_HISTORY_LEN,
    DEFAULT_EMERGENCY_DISTANCE_PX,
)
from src.utils.hsv import bgr_to_hsv_normalized

from .emergency_vehicle import EmergencyVehicleDetector, EmergencyVehicle, EmergencyVehicleType
from .feature import Direction
from .lane_violation import LaneViolationAnalyzer, build_analyzer_from_violation_config

logger = logging.getLogger(__name__)


class ViolationType(Enum):
    """违规类型"""
    RED_LIGHT = "red_light"      # 闯红灯
    SPEEDING = "speeding"        # 超速
    WRONG_WAY = "wrong_way"      # 逆行
    ILLEGAL_LANE = "illegal_lane"  # 违规变道


class AnomalyReason(Enum):
    """异常原因"""
    EMERGENCY_VEHICLE = "emergency_vehicle"  # 附近有特种车辆
    TRAFFIC_POLICE = "traffic_police"        # 附近有交警
    SIGNAL_MALFUNCTION = "signal_malfunction"  # 信号灯故障
    NONE = "none"


# 异常原因的中文描述
ANOMALY_DESCRIPTIONS = {
    AnomalyReason.EMERGENCY_VEHICLE: "附近有特种车辆",
    AnomalyReason.TRAFFIC_POLICE: "附近有交警指挥",
    AnomalyReason.SIGNAL_MALFUNCTION: "信号灯故障",
    AnomalyReason.NONE: "无",
}


@dataclass
class ViolationRecord:
    """违规记录"""
    record_id: str                          # 记录ID（时间戳）
    violation_type: ViolationType           # 违规类型
    track_id: int                           # 车辆跟踪ID
    timestamp: datetime                     # 发生时间
    location: Tuple[int, int]               # 位置
    speed: Optional[float] = None           # 速度
    plate_number: Optional[str] = None      # 车牌号
    snapshot_path: Optional[str] = None     # 截图路径
    is_anomaly: bool = False                # 是否为异常（需人工复核）
    anomaly_reason: AnomalyReason = AnomalyReason.NONE  # 异常原因
    nearby_objects: List[str] = field(default_factory=list)  # 附近特殊对象

    # ---- Backward compatibility for older callers ----
    @property
    def is_exempted(self) -> bool:
        return self.is_anomaly

    @is_exempted.setter
    def is_exempted(self, value: bool):
        self.is_anomaly = bool(value)

    @property
    def exemption_reason(self) -> AnomalyReason:
        return self.anomaly_reason

    @exemption_reason.setter
    def exemption_reason(self, value: AnomalyReason):
        self.anomaly_reason = value

    @property
    def nearby_emergency_vehicles(self) -> List[str]:
        return self.nearby_objects

    @nearby_emergency_vehicles.setter
    def nearby_emergency_vehicles(self, value: Optional[List[str]]):
        self.nearby_objects = list(value or [])

    @property
    def exemption_details(self) -> str:
        return ", ".join(self.nearby_objects)

    @exemption_details.setter
    def exemption_details(self, value: Optional[str]):
        if not value:
            self.nearby_objects = []
        else:
            self.nearby_objects = [item.strip() for item in value.split(",") if item.strip()]


# 兼容旧代码
ExemptionReason = AnomalyReason
EXEMPTION_DESCRIPTIONS = ANOMALY_DESCRIPTIONS


class TrafficPoliceDetector:
    """交警检测器（基于颜色和姿态特征）"""

    def __init__(self):
        # 交警制服颜色特征 (HSV) - 深蓝色/黑色
        self.uniform_lower = np.array([100, 50, 30])
        self.uniform_upper = np.array([130, 255, 150])
        # 反光背心颜色 - 荧光黄/绿
        self.vest_lower = np.array([25, 100, 100])
        self.vest_upper = np.array([45, 255, 255])

    def detect(self, frame: np.ndarray,
               person_bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        检测交警

        Args:
            frame: BGR图像
            person_bboxes: 人员边界框列表

        Returns:
            交警边界框列表
        """
        police_bboxes = []

        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            hsv = bgr_to_hsv_normalized(roi)
            total = roi.shape[0] * roi.shape[1]

            vest_mask = cv2.inRange(hsv, self.vest_lower, self.vest_upper)
            vest_ratio = np.sum(vest_mask > 0) / total

            # 检测制服颜色
            uniform_mask = cv2.inRange(hsv, self.uniform_lower, self.uniform_upper)
            uniform_ratio = np.sum(uniform_mask > 0) / total

            # 如果有反光背心或制服特征，认为是交警
            if vest_ratio > COLOR_RATIO_THRESHOLD or uniform_ratio > COLOR_RATIO_THRESHOLD * 2:
                police_bboxes.append(bbox)

        return police_bboxes


class TrafficLightDetector:
    """交通灯状态检测器"""

    def __init__(self):
        self.red_lower1 = np.array([0, 100, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([180, 255, 255])
        self.green_lower = np.array([35, 100, 100])
        self.green_upper = np.array([85, 255, 255])
        self.yellow_lower = np.array([15, 100, 100])
        self.yellow_upper = np.array([35, 255, 255])
        self.state_history = deque(maxlen=30)

    def detect_state(self, frame: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> str:
        """检测交通灯状态（HSV + 竖长外形 + 上中下灯珠分区）"""
        fh, fw = frame.shape[:2]
        clamped = clamp_bbox(bbox, fw, fh)
        if clamped is None:
            return 'unknown'
        x1, y1, x2, y2 = clamped
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 'unknown'

        aspect = (x2 - x1) / max(1, (y2 - y1))
        if not (TRAFFIC_LIGHT_MIN_ASPECT <= aspect <= TRAFFIC_LIGHT_MAX_ASPECT):
            return self._color_vote(roi)

        hsv = bgr_to_hsv_normalized(roi)
        h = roi.shape[0]
        bands = [
            hsv[0: max(1, h // 3), :],
            hsv[max(1, h // 3): max(2, 2 * h // 3), :],
            hsv[max(2, 2 * h // 3):, :],
        ]
        band_states = []
        for band in bands:
            if band.size == 0:
                continue
            band_states.append(self._dominant_color_on_band(band))

        if band_states:
            from collections import Counter
            state = Counter(band_states).most_common(1)[0][0]
        else:
            state = self._color_vote(roi)

        self.state_history.append(state)
        return state

    def _dominant_color_on_band(self, hsv_band: np.ndarray) -> str:
        total = hsv_band.shape[0] * hsv_band.shape[1]
        if total <= 0:
            return 'unknown'
        red_mask = (
            cv2.inRange(hsv_band, self.red_lower1, self.red_upper1)
            | cv2.inRange(hsv_band, self.red_lower2, self.red_upper2)
        )
        green_mask = cv2.inRange(hsv_band, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv_band, self.yellow_lower, self.yellow_upper)
        ratios = {
            'red': np.sum(red_mask > 0) / total,
            'green': np.sum(green_mask > 0) / total,
            'yellow': np.sum(yellow_mask > 0) / total,
        }
        best = max(ratios, key=ratios.get)
        if ratios[best] < TRAFFIC_LIGHT_COLOR_THRESHOLD:
            return 'unknown'
        return best

    def _color_vote(self, roi: np.ndarray) -> str:
        """整 ROI 颜色投票（无竖长外形时的回退）。"""
        hsv = bgr_to_hsv_normalized(roi)
        total = roi.shape[0] * roi.shape[1]
        if total <= 0:
            return 'unknown'
        red_mask = (
            cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            | cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        )
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        ratios = {
            'red': np.sum(red_mask > 0) / total,
            'green': np.sum(green_mask > 0) / total,
            'yellow': np.sum(yellow_mask > 0) / total,
        }
        best = max(ratios, key=ratios.get)
        if ratios[best] < TRAFFIC_LIGHT_COLOR_THRESHOLD:
            return 'unknown'
        self.state_history.append(best)
        return best

    def is_malfunctioning(self) -> bool:
        """检测信号灯是否故障"""
        if len(self.state_history) < 10:
            return False
        recent = list(self.state_history)[-10:]
        unknown_count = recent.count('unknown')
        if unknown_count >= 8:
            return True
        changes = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i-1])
        if changes >= 6:
            return True
        return False


class StopLine:
    """停止线"""

    def __init__(self, y: int, x_start: int, x_end: int):
        self.y = y
        self.x_start = x_start
        self.x_end = x_end

    def is_crossed(self, prev_center: Tuple[int, int],
                   curr_center: Tuple[int, int]) -> bool:
        """判断是否越过停止线"""
        if not (self.x_start <= curr_center[0] <= self.x_end):
            return False
        if prev_center[1] < self.y <= curr_center[1]:
            return True
        return False


class AdaptiveViolationDetector:
    """
    自适应违规检测器

    创新点：
    1. 检测特种车辆和交警
    2. 违规时如果附近有特种车辆或交警，标记为"异常"
    3. 异常情况单独保存，便于人工复核
    """

    def __init__(self,
                 speed_limit: float = 60.0,
                 stop_line: Optional[StopLine] = None,
                 snapshot_dir: str = "data/snapshots",
                 emergency_distance: int = DEFAULT_EMERGENCY_DISTANCE_PX,
                 wrong_way_enabled: bool = True,
                 illegal_lane_enabled: bool = True,
                 expected_flow_direction: str = "south",
                 lane_change_lateral_px: int = DEFAULT_LANE_CHANGE_LATERAL_PX,
                 lane_change_min_speed_kmh: float = DEFAULT_LANE_CHANGE_MIN_SPEED_KMH):
        self.speed_limit = speed_limit
        self.stop_line = stop_line
        self.snapshot_dir = Path(snapshot_dir)
        self.emergency_distance = emergency_distance
        self.wrong_way_enabled = wrong_way_enabled
        self.illegal_lane_enabled = illegal_lane_enabled
        self.expected_flow_direction = expected_flow_direction.lower()
        self.lane_change_lateral_px = lane_change_lateral_px
        self.lane_change_min_speed_kmh = lane_change_min_speed_kmh
        self.track_history_maxlen = DEFAULT_LANE_CHANGE_HISTORY_LEN
        self.lane_analyzer = build_analyzer_from_violation_config(
            expected_flow_direction=self.expected_flow_direction,
            lane_change_lateral_px=lane_change_lateral_px,
            lane_change_min_speed_kmh=lane_change_min_speed_kmh,
            history_len=self.track_history_maxlen,
        )

        # 创建截图目录
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        (self.snapshot_dir / "violations").mkdir(exist_ok=True)
        (self.snapshot_dir / "anomaly").mkdir(exist_ok=True)

        # 子模块
        self.traffic_light_detector = TrafficLightDetector()
        self.emergency_detector = EmergencyVehicleDetector()
        self.police_detector = TrafficPoliceDetector()

        # 状态
        self.track_history: Dict[int, deque] = {}
        self.recorded_violations: Dict[int, set] = {}
        self.current_light_state = 'unknown'
        self.current_emergency_vehicles: List[EmergencyVehicle] = []
        self.current_police_bboxes: List[Tuple[int, int, int, int]] = []

        # 统计
        self.total_violations = 0
        self.anomaly_count = 0

    def set_stop_line(self, y: int, x_start: int, x_end: int):
        """设置停止线"""
        self.stop_line = StopLine(y, x_start, x_end)

    def update(self, frame: np.ndarray,
               vehicle_bboxes: List[Tuple[int, int, int, int]],
               person_bboxes: List[Tuple[int, int, int, int]] = None,
               light_bbox: Optional[Tuple[int, int, int, int]] = None):
        """更新检测器状态"""
        if light_bbox:
            self.current_light_state = self.traffic_light_detector.detect_state(
                frame, light_bbox
            )

        # 检测特种车辆
        self.current_emergency_vehicles = self.emergency_detector.detect(
            frame, vehicle_bboxes
        )

        # 检测交警
        if person_bboxes:
            self.current_police_bboxes = self.police_detector.detect(
                frame, person_bboxes
            )
        else:
            self.current_police_bboxes = []

    def check_violation(self,
                        track_id: int,
                        bbox: Tuple[int, int, int, int],
                        speed: float,
                        frame: np.ndarray,
                        plate_number: Optional[str] = None,
                        direction: Optional[Direction] = None) -> Optional[ViolationRecord]:
        """检查单个车辆的违规行为"""
        fh, fw = frame.shape[:2]
        clamped = clamp_bbox(bbox, fw, fh)
        if clamped is None:
            return None
        x1, y1, x2, y2 = clamped
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.track_history_maxlen)
            self.recorded_violations[track_id] = set()

        history = self.track_history[track_id]

        violation_type = None
        violation_speed = None

        if speed > self.speed_limit:
            if ViolationType.SPEEDING not in self.recorded_violations[track_id]:
                violation_type = ViolationType.SPEEDING
                violation_speed = speed

        if (violation_type is None and
            self.stop_line and
            len(history) > 0 and
            self.current_light_state == 'red'):
            prev_center = history[-1]
            if self.stop_line.is_crossed(prev_center, center):
                if ViolationType.RED_LIGHT not in self.recorded_violations[track_id]:
                    violation_type = ViolationType.RED_LIGHT

        trajectory = list(history) + [center]
        lane_result = self.lane_analyzer.analyze(
            trajectory,
            speed_kmh=speed,
            direction=direction,
            wrong_way_enabled=self.wrong_way_enabled,
            illegal_lane_enabled=self.illegal_lane_enabled,
        )

        if (
            violation_type is None
            and lane_result.is_wrong_way
            and ViolationType.WRONG_WAY not in self.recorded_violations[track_id]
        ):
            violation_type = ViolationType.WRONG_WAY
            violation_speed = speed

        if (
            violation_type is None
            and lane_result.is_illegal_lane_change
            and ViolationType.ILLEGAL_LANE not in self.recorded_violations[track_id]
        ):
            violation_type = ViolationType.ILLEGAL_LANE
            violation_speed = speed

        history.append(center)

        if violation_type is None:
            return None

        # 检查是否为异常情况
        is_anomaly, anomaly_reason, nearby_objects = self._check_anomaly(bbox)

        # 生成记录
        timestamp = datetime.now()
        record_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        # 保存截图
        snapshot_path = self._save_snapshot(
            frame, record_id, is_anomaly, bbox, violation_type
        )

        record = ViolationRecord(
            record_id=record_id,
            violation_type=violation_type,
            track_id=track_id,
            timestamp=timestamp,
            location=center,
            speed=violation_speed,
            plate_number=plate_number,
            snapshot_path=snapshot_path,
            is_anomaly=is_anomaly,
            anomaly_reason=anomaly_reason,
            nearby_objects=nearby_objects
        )

        # 更新统计
        self.total_violations += 1
        if is_anomaly:
            self.anomaly_count += 1

        self.recorded_violations[track_id].add(violation_type)
        return record

    def _check_anomaly(self, vehicle_bbox: Tuple[int, int, int, int]
                       ) -> Tuple[bool, AnomalyReason, List[str]]:
        """
        检查是否为异常情况（附近有特种车辆或交警）

        Returns:
            (是否异常, 异常原因, 附近特殊对象列表)
        """
        nearby_objects = []
        vx = (vehicle_bbox[0] + vehicle_bbox[2]) // 2
        vy = (vehicle_bbox[1] + vehicle_bbox[3]) // 2

        # 1. 检查附近是否有特种车辆
        for ev in self.current_emergency_vehicles:
            ex = (ev.bbox[0] + ev.bbox[2]) // 2
            ey = (ev.bbox[1] + ev.bbox[3]) // 2
            distance = np.sqrt((vx - ex)**2 + (vy - ey)**2)

            if distance < self.emergency_distance:
                ev_name = self._get_emergency_vehicle_name(ev.vehicle_type)
                nearby_objects.append(ev_name)

        if nearby_objects:
            return True, AnomalyReason.EMERGENCY_VEHICLE, nearby_objects

        # 2. 检查附近是否有交警
        for police_bbox in self.current_police_bboxes:
            px = (police_bbox[0] + police_bbox[2]) // 2
            py = (police_bbox[1] + police_bbox[3]) // 2
            distance = np.sqrt((vx - px)**2 + (vy - py)**2)

            if distance < self.emergency_distance:
                nearby_objects.append("交警")

        if nearby_objects:
            return True, AnomalyReason.TRAFFIC_POLICE, nearby_objects

        # 3. 检查信号灯故障
        if self.traffic_light_detector.is_malfunctioning():
            return True, AnomalyReason.SIGNAL_MALFUNCTION, ["信号灯故障"]

        return False, AnomalyReason.NONE, []

    def _get_emergency_vehicle_name(self, ev_type: EmergencyVehicleType) -> str:
        """获取特种车辆中文名称"""
        names = {
            EmergencyVehicleType.AMBULANCE: "救护车",
            EmergencyVehicleType.FIRE_TRUCK: "消防车",
            EmergencyVehicleType.POLICE_CAR: "警车",
            EmergencyVehicleType.RESCUE_VEHICLE: "工程救险车",
            EmergencyVehicleType.UNKNOWN: "特种车辆",
        }
        return names.get(ev_type, "特种车辆")

    def _save_snapshot(self, frame: np.ndarray, record_id: str,
                       is_anomaly: bool, bbox: Tuple[int, int, int, int],
                       violation_type: ViolationType) -> str:
        """保存截图"""
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox

        # 黄色=异常待复核，红色=正常违规
        color = (0, 255, 255) if is_anomaly else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        label = f"{violation_type.value}"
        if is_anomaly:
            label += " [ANOMALY]"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, time_str, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        subdir = "anomaly" if is_anomaly else "violations"
        filename = f"{record_id}_{violation_type.value}.jpg"
        filepath = self.snapshot_dir / subdir / filename

        try:
            cv2.imwrite(str(filepath), annotated)
        except cv2.error as exc:
            logger.error("Failed to save snapshot %s: %s", filepath, exc)
            return ""
        return str(filepath)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_violations": self.total_violations,
            "anomaly_count": self.anomaly_count,
            "normal_violations": self.total_violations - self.anomaly_count,
            # 兼容旧代码
            "exempted_count": self.anomaly_count,
            "actual_violations": self.total_violations - self.anomaly_count,
            "exemption_rate": self.anomaly_count / max(1, self.total_violations),
        }

    def clear_track(self, track_id: int):
        """清除跟踪记录"""
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.recorded_violations:
            del self.recorded_violations[track_id]

    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """在帧上绘制标注"""
        annotated = frame.copy()

        # 绘制停止线
        if self.stop_line:
            color = (0, 0, 255) if self.current_light_state == 'red' else (0, 255, 0)
            cv2.line(annotated,
                     (self.stop_line.x_start, self.stop_line.y),
                     (self.stop_line.x_end, self.stop_line.y),
                     color, 2)

        # 绘制交通灯状态
        light_colors = {'red': (0, 0, 255), 'green': (0, 255, 0),
                        'yellow': (0, 255, 255), 'unknown': (128, 128, 128)}
        cv2.circle(annotated, (30, 60), 15,
                   light_colors.get(self.current_light_state, (128, 128, 128)), -1)

        # 标注特种车辆
        for ev in self.current_emergency_vehicles:
            x1, y1, x2, y2 = ev.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # OpenCV 默认字体不支持中文，这里用英文缩写避免出现“????”
            ev_labels_en = {
                EmergencyVehicleType.AMBULANCE: "Ambulance",
                EmergencyVehicleType.FIRE_TRUCK: "Fire Truck",
                EmergencyVehicleType.POLICE_CAR: "Police Car",
                EmergencyVehicleType.RESCUE_VEHICLE: "Rescue",
                EmergencyVehicleType.UNKNOWN: "Emergency",
            }
            label_en = ev_labels_en.get(ev.vehicle_type, "Emergency")
            cv2.putText(annotated, label_en, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # 标注交警
        for bbox in self.current_police_bboxes:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 3)
            cv2.putText(annotated, "Traffic Police", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

        # 显示统计
        stats = self.get_statistics()
        info = f"Violations: {stats['normal_violations']} | Anomaly: {stats['anomaly_count']}"
        cv2.putText(annotated, info, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated
