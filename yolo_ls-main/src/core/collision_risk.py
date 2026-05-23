"""
Collision Risk Prediction Module
Predicts future trajectories and assesses collision risks

Innovation: Uses LSTM-based trajectory prediction combined with
Time-To-Collision (TTC) analysis for proactive safety warnings.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque
from pathlib import Path
import cv2


class RiskLevel(Enum):
    """Collision risk levels"""
    SAFE = "safe"           # No risk
    LOW = "low"             # Monitor
    MEDIUM = "medium"       # Warning
    HIGH = "high"           # Danger
    CRITICAL = "critical"   # Imminent collision


@dataclass
class CollisionRisk:
    """Collision risk assessment result"""
    vehicle1_id: int
    vehicle2_id: int
    risk_level: RiskLevel
    time_to_collision: float  # seconds, -1 if no collision predicted
    collision_point: Optional[Tuple[float, float]]
    confidence: float
    predicted_trajectories: Dict[int, List[Tuple[float, float]]]


class TrajectoryPredictor(nn.Module):
    """
    LSTM-based trajectory prediction network
    Predicts future positions based on historical trajectory
    """
    def __init__(self,
                 input_dim: int = 4,      # (x, y, vx, vy)
                 hidden_dim: int = 64,
                 output_dim: int = 2,     # (x, y)
                 num_layers: int = 2,
                 pred_horizon: int = 10): # Predict 10 future steps
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_horizon = pred_horizon

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder for multi-step prediction
        self.decoder = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Historical trajectory [batch, seq_len, input_dim]
        Returns:
            Predicted future positions [batch, pred_horizon, 2]
        """
        batch_size = x.size(0)

        # Encode history
        _, (h, c) = self.encoder(x)

        # Decode future
        # Start with last known position
        decoder_input = x[:, -1, :2].unsqueeze(1)  # [batch, 1, 2]
        predictions = []

        for _ in range(self.pred_horizon):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)  # [batch, 1, 2]
            predictions.append(pred)
            decoder_input = pred

        return torch.cat(predictions, dim=1)  # [batch, pred_horizon, 2]


class CollisionRiskPredictor:
    """
    Collision Risk Prediction System

    Features:
    1. LSTM-based trajectory prediction
    2. Time-To-Collision (TTC) calculation
    3. Multi-level risk assessment
    4. Proactive warning generation
    """
    def __init__(self,
                 history_length: int = 10,
                 prediction_horizon: int = 15,
                 fps: float = 15.0,
                 collision_threshold: float = 150.0,  # 增大默认阈值适应高分辨率
                 ttc_thresholds: Dict[str, float] = None,
                 model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Args:
            history_length: Number of historical frames to use
            prediction_horizon: Number of future frames to predict
            fps: Video frame rate
            collision_threshold: Distance threshold for collision (pixels)
            ttc_thresholds: Time-to-collision thresholds for risk levels
            model_path: Optional trained trajectory model path
            device: cpu/cuda
        """
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.fps = fps
        self.collision_threshold = collision_threshold
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Default TTC thresholds (in seconds)
        self.ttc_thresholds = ttc_thresholds or {
            'critical': 0.5,
            'high': 1.0,
            'medium': 2.0,
            'low': 3.0
        }

        # Track history
        self.track_history: Dict[int, deque] = {}

        # Trajectory predictor
        self.predictor = TrajectoryPredictor(
            input_dim=4,
            hidden_dim=64,
            pred_horizon=prediction_horizon
        ).to(self.device)

        if model_path:
            self.load_model(model_path)

        self.predictor.eval()

    def load_model(self, model_path: str) -> bool:
        """Load trained trajectory predictor checkpoint."""
        path = Path(model_path)
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.predictor.load_state_dict(state_dict, strict=False)
        self.predictor.eval()
        return True

    def save_model(self, model_path: str, extra: Optional[Dict[str, Any]] = None):
        """Save trajectory predictor checkpoint."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            'state_dict': self.predictor.state_dict(),
            'history_length': self.history_length,
            'prediction_horizon': self.prediction_horizon,
            'fps': self.fps,
            'collision_threshold': self.collision_threshold,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def update(self, tracks: List[Dict]) -> List[CollisionRisk]:
        """
        Update with current frame and predict collision risks

        Args:
            tracks: List of track dicts with bbox and track_id

        Returns:
            List of collision risk assessments
        """
        # Update history
        for track in tracks:
            tid = track['track_id']
            bbox = track['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=self.history_length)

            # Compute velocity
            vx, vy = 0.0, 0.0
            if len(self.track_history[tid]) > 0:
                prev = self.track_history[tid][-1]
                vx = cx - prev[0]
                vy = cy - prev[1]

            self.track_history[tid].append((cx, cy, vx, vy))

        # Clean old tracks
        current_ids = {t['track_id'] for t in tracks}
        self.track_history = {
            tid: hist for tid, hist in self.track_history.items()
            if tid in current_ids
        }

        # Predict trajectories for vehicles with enough history
        predictions = {}
        for tid, history in self.track_history.items():
            if len(history) >= 5:  # Need at least 5 frames
                pred = self._predict_trajectory(tid)
                if pred is not None:
                    predictions[tid] = pred

        # Assess collision risks between all pairs
        risks = []
        track_ids = list(predictions.keys())
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                # 1. 检查轨迹预测碰撞风险
                risk = self._assess_collision_risk(
                    track_ids[i], track_ids[j], predictions
                )
                if risk.risk_level != RiskLevel.SAFE:
                    risks.append(risk)
                else:
                    # 2. 检查跟车距离风险
                    following_risk = self._assess_following_distance(
                        track_ids[i], track_ids[j], predictions
                    )
                    if following_risk and following_risk.risk_level != RiskLevel.SAFE:
                        risks.append(following_risk)

        # Sort by risk level
        risk_order = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3,
            RiskLevel.SAFE: 4
        }
        risks.sort(key=lambda r: risk_order[r.risk_level])

        return risks

    def _predict_trajectory(self, track_id: int) -> Optional[List[Tuple[float, float]]]:
        """Predict future trajectory for a vehicle"""
        history = list(self.track_history[track_id])

        if len(history) < 5:
            return None

        # Prepare input
        input_seq = np.array(history[-self.history_length:], dtype=np.float32)

        # Normalize
        mean_pos = input_seq[:, :2].mean(axis=0)
        input_seq[:, :2] -= mean_pos

        # Predict
        with torch.no_grad():
            x = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
            pred = self.predictor(x).squeeze(0).cpu().numpy()

        # Denormalize
        pred += mean_pos

        return [(float(p[0]), float(p[1])) for p in pred]

    def _assess_collision_risk(self,
                                id1: int,
                                id2: int,
                                predictions: Dict[int, List[Tuple[float, float]]]) -> CollisionRisk:
        """Assess collision risk between two vehicles"""
        traj1 = predictions[id1]
        traj2 = predictions[id2]

        # Find minimum distance and time
        min_dist = float('inf')
        min_time_idx = -1
        collision_point = None

        for t in range(min(len(traj1), len(traj2))):
            dist = np.sqrt(
                (traj1[t][0] - traj2[t][0])**2 +
                (traj1[t][1] - traj2[t][1])**2
            )
            if dist < min_dist:
                min_dist = dist
                min_time_idx = t
                collision_point = (
                    (traj1[t][0] + traj2[t][0]) / 2,
                    (traj1[t][1] + traj2[t][1]) / 2
                )

        # Calculate TTC
        if min_dist < self.collision_threshold:
            ttc = min_time_idx / self.fps
        else:
            ttc = -1  # No collision predicted

        # Determine risk level
        risk_level = self._get_risk_level(ttc, min_dist)

        # Calculate confidence based on trajectory smoothness
        confidence = self._calculate_confidence(id1, id2)

        return CollisionRisk(
            vehicle1_id=id1,
            vehicle2_id=id2,
            risk_level=risk_level,
            time_to_collision=ttc,
            collision_point=collision_point if ttc > 0 else None,
            confidence=confidence,
            predicted_trajectories={id1: traj1, id2: traj2}
        )

    def _get_risk_level(self, ttc: float, min_dist: float) -> RiskLevel:
        """Determine risk level based on TTC and distance"""
        if ttc < 0:
            # No collision predicted
            if min_dist < self.collision_threshold * 2:
                return RiskLevel.LOW
            return RiskLevel.SAFE

        if ttc <= self.ttc_thresholds['critical']:
            return RiskLevel.CRITICAL
        elif ttc <= self.ttc_thresholds['high']:
            return RiskLevel.HIGH
        elif ttc <= self.ttc_thresholds['medium']:
            return RiskLevel.MEDIUM
        elif ttc <= self.ttc_thresholds['low']:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE

    def _assess_following_distance(self,
                                    id1: int,
                                    id2: int,
                                    predictions: Dict[int, List[Tuple[float, float]]]) -> Optional[CollisionRisk]:
        """
        评估跟车距离风险
        当两车同向行驶且距离过近时触发
        """
        if id1 not in self.track_history or id2 not in self.track_history:
            return None

        hist1 = list(self.track_history[id1])
        hist2 = list(self.track_history[id2])

        if len(hist1) < 3 or len(hist2) < 3:
            return None

        # 获取当前位置和速度
        pos1 = (hist1[-1][0], hist1[-1][1])
        pos2 = (hist2[-1][0], hist2[-1][1])
        vel1 = (hist1[-1][2], hist1[-1][3])
        vel2 = (hist2[-1][2], hist2[-1][3])

        # 计算速度大小
        speed1 = np.sqrt(vel1[0]**2 + vel1[1]**2)
        speed2 = np.sqrt(vel2[0]**2 + vel2[1]**2)

        # 计算速度方向（角度）
        if speed1 < 1 or speed2 < 1:  # 速度太小，忽略
            return None

        angle1 = np.arctan2(vel1[1], vel1[0])
        angle2 = np.arctan2(vel2[1], vel2[0])

        # 检查是否同向行驶（角度差小于30度）
        angle_diff = abs(angle1 - angle2)
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        if angle_diff > np.pi / 6:  # 30度
            return None  # 不是同向行驶

        # 计算当前距离
        current_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        # 计算安全跟车距离（基于速度）
        # 安全距离 = 速度 * 反应时间系数 + 最小安全距离
        avg_speed = (speed1 + speed2) / 2
        safe_distance = avg_speed * 3.0 + 100  # 3秒反应时间 + 100像素最小距离（适应高分辨率）

        # 判断风险等级
        if current_dist < safe_distance * 0.3:
            risk_level = RiskLevel.CRITICAL
        elif current_dist < safe_distance * 0.5:
            risk_level = RiskLevel.HIGH
        elif current_dist < safe_distance * 0.7:
            risk_level = RiskLevel.MEDIUM
        elif current_dist < safe_distance * 1.0:
            risk_level = RiskLevel.LOW
        else:
            return None  # 安全

        # 计算TTC（基于相对速度）
        # 确定前后车
        dir_vec = (pos2[0] - pos1[0], pos2[1] - pos1[1])
        dot_product = dir_vec[0] * vel1[0] + dir_vec[1] * vel1[1]

        if dot_product > 0:
            # id1在后，id2在前
            relative_speed = speed1 - speed2
        else:
            # id2在后，id1在前
            relative_speed = speed2 - speed1

        if relative_speed > 0.5:  # 后车更快
            ttc = current_dist / relative_speed / self.fps
        else:
            ttc = -1  # 不会追上

        return CollisionRisk(
            vehicle1_id=id1,
            vehicle2_id=id2,
            risk_level=risk_level,
            time_to_collision=ttc,
            collision_point=((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2),
            confidence=0.7,
            predicted_trajectories=predictions
        )

    def _calculate_confidence(self, id1: int, id2: int) -> float:
        """Calculate prediction confidence based on trajectory smoothness"""
        confidence = 0.5

        for tid in [id1, id2]:
            if tid in self.track_history:
                history = list(self.track_history[tid])
                if len(history) >= 3:
                    # Check velocity consistency
                    velocities = [(h[2], h[3]) for h in history[-5:]]
                    if len(velocities) >= 2:
                        var = np.var([np.sqrt(v[0]**2 + v[1]**2) for v in velocities])
                        # Lower variance = higher confidence
                        confidence += 0.25 * (1 / (1 + var))

        return min(confidence, 1.0)

    def get_vehicle_risk_colors(self, risks: List[CollisionRisk]) -> Dict[int, Tuple[int, int, int]]:
        """
        获取每个车辆的风险颜色
        Returns:
            Dict mapping track_id to BGR color tuple
        """
        colors = {
            RiskLevel.SAFE: (0, 255, 0),      # Green
            RiskLevel.LOW: (0, 255, 255),     # Yellow
            RiskLevel.MEDIUM: (0, 165, 255),  # Orange
            RiskLevel.HIGH: (0, 0, 255),      # Red
            RiskLevel.CRITICAL: (255, 0, 255) # Magenta
        }

        vehicle_colors: Dict[int, Tuple[int, int, int]] = {}

        for risk in risks:
            color = colors[risk.risk_level]
            # 为涉及风险的两辆车都设置颜色（取最高风险等级）
            for vid in [risk.vehicle1_id, risk.vehicle2_id]:
                if vid not in vehicle_colors:
                    vehicle_colors[vid] = color
                else:
                    # 如果已有颜色，保留更高风险的颜色
                    existing_risk = self._color_to_risk_level(vehicle_colors[vid], colors)
                    new_risk = risk.risk_level
                    if self._risk_level_priority(new_risk) < self._risk_level_priority(existing_risk):
                        vehicle_colors[vid] = color

        return vehicle_colors

    def _color_to_risk_level(self, color: Tuple[int, int, int],
                              colors: Dict[RiskLevel, Tuple[int, int, int]]) -> RiskLevel:
        """根据颜色反查风险等级"""
        for level, c in colors.items():
            if c == color:
                return level
        return RiskLevel.SAFE

    def _risk_level_priority(self, level: RiskLevel) -> int:
        """风险等级优先级（数字越小优先级越高）"""
        priority = {
            RiskLevel.CRITICAL: 0,
            RiskLevel.HIGH: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 3,
            RiskLevel.SAFE: 4
        }
        return priority.get(level, 4)

    def draw_predictions(self, frame: np.ndarray,
                         risks: List[CollisionRisk],
                         tracks: List[Dict] = None) -> np.ndarray:
        """
        Draw collision risk indicators on frame
        Args:
            frame: Input frame
            risks: List of collision risks
            tracks: Optional list of track dicts with bbox for drawing colored boxes
        """
        annotated = frame.copy()

        colors = {
            RiskLevel.SAFE: (0, 255, 0),      # Green
            RiskLevel.LOW: (0, 255, 255),     # Yellow
            RiskLevel.MEDIUM: (0, 165, 255),  # Orange
            RiskLevel.HIGH: (0, 0, 255),      # Red
            RiskLevel.CRITICAL: (255, 0, 255) # Magenta
        }

        # 获取每个车辆的风险颜色
        vehicle_colors = self.get_vehicle_risk_colors(risks)

        # 如果提供了 tracks，为有风险的车辆画彩色边框
        if tracks:
            for track in tracks:
                tid = track['track_id']
                bbox = track['bbox']
                if tid in vehicle_colors:
                    color = vehicle_colors[tid]
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    # 画粗边框表示风险
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                    # 画内边框增强效果
                    cv2.rectangle(annotated, (x1+2, y1+2), (x2-2, y2-2), color, 2)

        # 绘制预测轨迹线（可选，用虚线表示）
        for risk in risks:
            if risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                color = colors[risk.risk_level]
                for tid, traj in risk.predicted_trajectories.items():
                    points = [(int(p[0]), int(p[1])) for p in traj]
                    for i in range(1, len(points), 2):  # 虚线效果
                        alpha = 1 - (i / len(points))
                        cv2.line(annotated, points[i-1], points[i],
                                 tuple(int(c * alpha) for c in color), 2)

                # 绘制碰撞点（小圆点）
                if risk.collision_point:
                    cp = (int(risk.collision_point[0]), int(risk.collision_point[1]))
                    cv2.circle(annotated, cp, 8, color, -1)
                    if risk.time_to_collision > 0:
                        text = f"TTC:{risk.time_to_collision:.1f}s"
                        cv2.putText(annotated, text, (cp[0] + 10, cp[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 绘制风险摘要
        if risks:
            highest_risk = risks[0]
            text = f"Risk: {highest_risk.risk_level.value.upper()}"
            cv2.putText(annotated, text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        colors[highest_risk.risk_level], 2)

        return annotated

    def get_risk_summary(self, risks: List[CollisionRisk]) -> Dict:
        """Get summary of current risks"""
        summary = {
            'total_risks': len(risks),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'min_ttc': float('inf'),
            'highest_risk_pair': None
        }

        for risk in risks:
            if risk.risk_level == RiskLevel.CRITICAL:
                summary['critical'] += 1
            elif risk.risk_level == RiskLevel.HIGH:
                summary['high'] += 1
            elif risk.risk_level == RiskLevel.MEDIUM:
                summary['medium'] += 1
            elif risk.risk_level == RiskLevel.LOW:
                summary['low'] += 1

            if risk.time_to_collision > 0 and risk.time_to_collision < summary['min_ttc']:
                summary['min_ttc'] = risk.time_to_collision
                summary['highest_risk_pair'] = (risk.vehicle1_id, risk.vehicle2_id)

        if summary['min_ttc'] == float('inf'):
            summary['min_ttc'] = -1

        return summary
