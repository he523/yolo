"""特种车辆检测模块检测救护车、消防车、警车等特种车辆"""
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EmergencyVehicleType(Enum):
    """特种车辆类型"""
    AMBULANCE = "ambulance"  # 救护车
    FIRE_TRUCK = "fire_truck"  # 消防车
    POLICE_CAR = "police_car"  # 警车
    RESCUE_VEHICLE = "rescue"  # 工程救险车
    UNKNOWN = "unknown"

@dataclass
class EmergencyVehicle:
    """特种车辆检测结果"""
    vehicle_type: EmergencyVehicleType
    bbox: Tuple[int, int, int, int]
    confidence: float
    has_warning_light: bool  # 是否有警示灯闪烁
    has_siren_color: bool  # 是否有特种车辆颜色特征

class EmergencyVehicleDetector:
    """
    特种车辆检测器
    通过以下特征识别特种车辆：
    1. 颜色特征（红色消防车、白色救护车、蓝白警车）
    2. 警示灯检测（红蓝闪烁灯）
    3. 车身标识（可选，需要OCR）
    """

    # 特种车辆颜色特征 (HSV范围)
    COLOR_FEATURES = {
        EmergencyVehicleType.FIRE_TRUCK: {
            'primary': [(0, 100, 100), (10, 255, 255)],  # 红色
            'secondary': [(160, 100, 100), (180, 255, 255)],  # 红色（另一范围）
        },
        EmergencyVehicleType.AMBULANCE: {
            'primary': [(0, 0, 200), (180, 30, 255)],  # 白色
        },
        EmergencyVehicleType.POLICE_CAR: {
            'primary': [(100, 100, 100), (130, 255, 255)],  # 蓝色
            'secondary': [(0, 0, 200), (180, 30, 255)],  # 白色
        },
    }

    # 警示灯颜色 (HSV范围)
    WARNING_LIGHT_COLORS = {
        'red': [(0, 150, 150), (10, 255, 255)],
        'red2': [(160, 150, 150), (180, 255, 255)],
        'blue': [(100, 150, 150), (130, 255, 255)],
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化检测器
        Args:
            config: 可选配置字典（来自 YAML emergency_vehicle 节）
        """
        cfg = config or {}
        self.min_light_area = cfg.get('min_light_area', 100)
        self._color_threshold = cfg.get('color_threshold', 0.08)
        self.prev_frame = None
        self.light_history = {}  # 用于检测闪烁

    def detect(self, frame: np.ndarray, vehicle_bboxes: List[Tuple[int, int, int, int]]) -> List[EmergencyVehicle]:
        """
        检测特种车辆
        Args:
            frame: BGR图像
            vehicle_bboxes: 已检测到的车辆边界框列表，格式为[(x1,y1,x2,y2),...]
        Returns:
            特种车辆列表
        """
        # 检查帧是否有效
        if frame is None or frame.size == 0:
            return []

        results = []
        frame_h, frame_w = frame.shape[:2]

        for bbox in vehicle_bboxes:
            # 验证边界框是否在图像范围内
            x1, y1, x2, y2 = bbox
            # 确保坐标是整数
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 修正边界框，确保在图像范围内
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))

            # 跳过无效的边界框
            if x2 <= x1 or y2 <= y1:
                continue

            # 使用修正后的边界框
            fixed_bbox = (x1, y1, x2, y2)
            vehicle = self._analyze_vehicle(frame, fixed_bbox)
            if vehicle is not None:
                results.append(vehicle)

        self.prev_frame = frame.copy()
        return results

    def _analyze_vehicle(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[EmergencyVehicle]:
        """分析单个车辆是否为特种车辆"""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]

        # 检查ROI是否有效
        if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return None

        # 检测警示灯
        has_warning_light, light_colors = self._detect_warning_lights(roi)

        # 检测车身颜色特征
        vehicle_type, color_confidence = self._detect_vehicle_color(roi)

        # 综合判断
        if has_warning_light or color_confidence > 0.3:
            # 根据警示灯颜色调整车辆类型判断
            if has_warning_light:
                if 'red' in light_colors and vehicle_type == EmergencyVehicleType.UNKNOWN:
                    vehicle_type = EmergencyVehicleType.FIRE_TRUCK
                elif 'blue' in light_colors and vehicle_type == EmergencyVehicleType.UNKNOWN:
                    vehicle_type = EmergencyVehicleType.POLICE_CAR

            if vehicle_type != EmergencyVehicleType.UNKNOWN:
                return EmergencyVehicle(
                    vehicle_type=vehicle_type,
                    bbox=bbox,
                    confidence=color_confidence,
                    has_warning_light=has_warning_light,
                    has_siren_color=color_confidence > 0.3
                )
        return None

    def _detect_warning_lights(self, roi: np.ndarray) -> Tuple[bool, List[str]]:
        """检测警示灯，通过检测红蓝高亮区域来识别警示灯"""
        # 检查ROI是否为空
        if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return False, []

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error as e:
            logger.warning("Error converting ROI to HSV: %s", e)
            return False, []

        detected_colors = []

        # 只检测车辆上半部分（警示灯通常在车顶）
        # 确保有足够高度进行分割
        if roi.shape[0] > 6:  # 至少要有6像素高度
            top_roi = hsv[:roi.shape[0]//3, :]

            # 检查top_roi是否有效
            if top_roi.size == 0 or top_roi.shape[0] == 0 or top_roi.shape[1] == 0:
                return False, []

            for color_name, ranges in self.WARNING_LIGHT_COLORS.items():
                if isinstance(ranges, tuple) and len(ranges) == 2:
                    lower, upper = ranges
                else:
                    # 处理可能的格式错误
                    continue

                try:
                    mask = cv2.inRange(top_roi, np.array(lower), np.array(upper))
                except cv2.error as e:
                    logger.warning("Error in cv2.inRange for color %s: %s", color_name, e)
                    continue

                # 形态学操作去噪
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

                # 检测高亮区域
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > self.min_light_area:
                        # 检查亮度（警示灯通常很亮）
                        x, y, w, h = cv2.boundingRect(cnt)
                        # 确保区域在范围内
                        y_end = min(y + h, top_roi.shape[0])
                        x_end = min(x + w, top_roi.shape[1])

                        if y_end <= y or x_end <= x:
                            continue

                        # 获取对应的BGR区域
                        light_region = roi[:roi.shape[0]//3, :][y:y_end, x:x_end]
                        if light_region.size > 0 and light_region.shape[0] > 0 and light_region.shape[1] > 0:
                            brightness = np.mean(light_region)
                            if brightness > 150:  # 高亮度阈值
                                base_color = color_name.replace('2', '')
                                if base_color not in detected_colors:
                                    detected_colors.append(base_color)

        has_warning = len(detected_colors) > 0
        return has_warning, detected_colors

    def _detect_vehicle_color(self, roi: np.ndarray) -> Tuple[EmergencyVehicleType, float]:
        """检测车身颜色特征"""
        # 检查ROI是否有效
        if roi is None or roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            return EmergencyVehicleType.UNKNOWN, 0.0

        try:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        except cv2.error:
            return EmergencyVehicleType.UNKNOWN, 0.0

        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return EmergencyVehicleType.UNKNOWN, 0.0

        best_type = EmergencyVehicleType.UNKNOWN
        best_confidence = 0.0

        for vehicle_type, color_ranges in self.COLOR_FEATURES.items():
            type_score = 0.0
            range_count = 0

            for range_name, ranges in color_ranges.items():
                if isinstance(ranges, tuple) and len(ranges) == 2:
                    lower, upper = ranges
                    range_count += 1
                    try:
                        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                        ratio = np.sum(mask > 0) / total_pixels
                        type_score += ratio
                    except cv2.error:
                        continue

            if range_count > 0:
                type_score /= range_count
                if type_score > best_confidence:
                    best_confidence = type_score
                    best_type = vehicle_type

        return best_type, best_confidence

    def is_emergency_vehicle_nearby(self, emergency_vehicles: List[EmergencyVehicle],
                                   vehicle_bbox: Tuple[int, int, int, int],
                                   distance_threshold: int = 200) -> bool:
        """
        判断指定车辆附近是否有特种车辆
        Args:
            emergency_vehicles: 检测到的特种车辆列表
            vehicle_bbox: 待判断车辆的边界框
            distance_threshold: 距离阈值（像素）
        Returns:
            是否有特种车辆在附近
        """
        if not emergency_vehicles:
            return False

        # 计算待判断车辆的中心点
        x1, y1, x2, y2 = vehicle_bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        for ev in emergency_vehicles:
            ex1, ey1, ex2, ey2 = ev.bbox
            ev_center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)

            # 计算距离
            distance = float(np.hypot(center[0] - ev_center[0], center[1] - ev_center[1]))
            if distance < distance_threshold:
                return True

        return False