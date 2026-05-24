"""Core modules"""
from .detector import VehicleDetector, Detection
from .tracker import ByteTracker, Track
from .feature import FeatureExtractor, VehicleFeatures, Direction, ColorAnalyzer, SpeedCalculator
# ViolationDetector / StopLine 已迁移至 adaptive_violation.py（旧 violation.py 已删除）
from .lane_violation import (
    LaneViolationAnalyzer,
    LaneViolationResult,
    WrongWayConfig,
    IllegalLaneChangeConfig,
    build_analyzer_from_violation_config,
)
from .emergency_vehicle import EmergencyVehicleDetector, EmergencyVehicle, EmergencyVehicleType
from .adaptive_violation import (
    AdaptiveViolationDetector, ViolationRecord, ExemptionReason,
    EXEMPTION_DESCRIPTIONS, ViolationType as AdaptiveViolationType,
    select_best_light_bbox,
)
from .stgat import VehicleInteractionGraph, SpatioTemporalGAT
from .collision_risk import CollisionRiskPredictor, CollisionRisk, RiskLevel

__all__ = [
    'VehicleDetector', 'Detection',
    'ByteTracker', 'Track',
    'FeatureExtractor', 'VehicleFeatures', 'Direction', 'ColorAnalyzer', 'SpeedCalculator',
    # 'ViolationDetector', 'Violation', 'ViolationType', 'StopLine',  — 已删除
    'LaneViolationAnalyzer', 'LaneViolationResult', 'WrongWayConfig',
    'IllegalLaneChangeConfig', 'build_analyzer_from_violation_config',
    'EmergencyVehicleDetector', 'EmergencyVehicle', 'EmergencyVehicleType',
    'AdaptiveViolationDetector', 'ViolationRecord', 'ExemptionReason',
    'EXEMPTION_DESCRIPTIONS', 'AdaptiveViolationType',
    'VehicleInteractionGraph', 'SpatioTemporalGAT',
    'CollisionRiskPredictor', 'CollisionRisk', 'RiskLevel'
]
