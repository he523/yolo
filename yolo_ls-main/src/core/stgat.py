"""
Spatio-Temporal Graph Attention Network (ST-GAT) Module
For modeling vehicle interactions and relationships in traffic scenes

Innovation: Uses graph attention to learn vehicle interaction patterns,
enabling better understanding of yielding behavior and collision risks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque
from pathlib import Path


@dataclass
class VehicleNode:
    """Vehicle node in the interaction graph"""
    track_id: int
    position: Tuple[float, float]  # (x, y) center
    velocity: Tuple[float, float]  # (vx, vy)
    bbox: Tuple[int, int, int, int]
    features: np.ndarray  # Appearance/motion features


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    Learns attention weights between vehicle nodes
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention mechanism
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        Returns:
            Updated node features [N, out_features]
        """
        N = h.size(0)

        # Linear transformation
        Wh = self.W(h)  # [N, out_features]

        # Compute attention coefficients
        # Concatenate all pairs of node features
        Wh_repeat_i = Wh.repeat(N, 1)  # [N*N, out_features]
        Wh_repeat_j = Wh.repeat_interleave(N, dim=0)  # [N*N, out_features]
        concat = torch.cat([Wh_repeat_i, Wh_repeat_j], dim=1)  # [N*N, 2*out_features]

        e = self.leaky_relu(self.a(concat)).view(N, N)  # [N, N]

        # Mask attention with adjacency matrix
        attention = torch.where(adj > 0, e, torch.tensor(-1e9).to(e.device))
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]

        return F.elu(h_prime)


class SpatioTemporalGAT(nn.Module):
    """
    Spatio-Temporal Graph Attention Network
    Combines spatial graph attention with temporal LSTM
    """
    def __init__(self,
                 node_features: int = 8,
                 hidden_dim: int = 32,
                 output_dim: int = 16,
                 num_heads: int = 2,
                 temporal_steps: int = 5):
        super().__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.temporal_steps = temporal_steps

        # Multi-head graph attention
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(node_features, hidden_dim)
            for _ in range(num_heads)
        ])

        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim * num_heads,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features: torch.Tensor,
                adj_matrix: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: [N, node_features]
            adj_matrix: [N, N]
            temporal_features: [N, T, hidden_dim] (optional)
        Returns:
            Node embeddings [N, output_dim]
        """
        # Multi-head attention
        head_outputs = [gat(node_features, adj_matrix) for gat in self.gat_layers]
        h = torch.cat(head_outputs, dim=1)  # [N, hidden_dim * num_heads]

        # If temporal features provided, use LSTM
        if temporal_features is not None:
            # Combine current with temporal
            h = h.unsqueeze(1)  # [N, 1, hidden_dim * num_heads]
            lstm_out, _ = self.lstm(h)
            h = lstm_out[:, -1, :]  # [N, hidden_dim]
        else:
            h = h[:, :self.hidden_dim]  # Take first hidden_dim features

        return self.fc(h)


class VehicleInteractionGraph:
    """
    Builds and manages the vehicle interaction graph
    """
    def __init__(self,
                 distance_threshold: float = 200.0,
                 temporal_window: int = 10,
                 model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Args:
            distance_threshold: Max distance for edge connection (pixels)
            temporal_window: Number of frames to keep in history
            model_path: Optional ST-GAT checkpoint path
            device: cpu/cuda
        """
        self.distance_threshold = distance_threshold
        self.temporal_window = temporal_window
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # Track history for each vehicle
        self.track_history: Dict[int, deque] = {}

        # ST-GAT model (lightweight, can run on CPU)
        self.model = SpatioTemporalGAT(
            node_features=8,
            hidden_dim=32,
            output_dim=16
        ).to(self.device)

        if model_path:
            self.load_model(model_path)

        self.model.eval()

    def load_model(self, model_path: str) -> bool:
        """Load trained ST-GAT checkpoint."""
        path = Path(model_path)
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        return True

    def save_model(self, model_path: str, extra: Optional[Dict[str, Any]] = None):
        """Save ST-GAT checkpoint."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            'state_dict': self.model.state_dict(),
            'distance_threshold': self.distance_threshold,
            'temporal_window': self.temporal_window,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def update(self, tracks: List[Dict]) -> Dict[int, np.ndarray]:
        """
        Update graph with current frame tracks

        Args:
            tracks: List of track dicts with keys:
                - track_id: int
                - bbox: (x1, y1, x2, y2)
                - center: (cx, cy)

        Returns:
            Dict mapping track_id to interaction embedding
        """
        if len(tracks) == 0:
            return {}

        # Update history
        for track in tracks:
            tid = track['track_id']
            if tid not in self.track_history:
                self.track_history[tid] = deque(maxlen=self.temporal_window)
            self.track_history[tid].append(track)

        # Clean old tracks
        current_ids = {t['track_id'] for t in tracks}
        old_ids = [tid for tid in self.track_history if tid not in current_ids]
        for tid in old_ids:
            if len(self.track_history[tid]) > 0:
                # Keep for a few frames in case of occlusion
                last_update = self.track_history[tid][-1].get('frame', 0)
                if len(self.track_history[tid]) >= self.temporal_window:
                    del self.track_history[tid]

        # Build node features
        node_features = self._build_node_features(tracks)

        # Build adjacency matrix
        adj_matrix = self._build_adjacency(tracks)

        # Run ST-GAT
        with torch.no_grad():
            node_features_t = torch.FloatTensor(node_features).to(self.device)
            adj_matrix_t = torch.FloatTensor(adj_matrix).to(self.device)
            embeddings = self.model(node_features_t, adj_matrix_t)
            embeddings = embeddings.cpu().numpy()

        # Map back to track IDs
        result = {}
        for i, track in enumerate(tracks):
            result[track['track_id']] = embeddings[i]

        return result

    def _build_node_features(self, tracks: List[Dict]) -> np.ndarray:
        """Build node feature matrix"""
        features = []
        for track in tracks:
            bbox = track['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            # Compute velocity from history
            tid = track['track_id']
            vx, vy = 0.0, 0.0
            if tid in self.track_history and len(self.track_history[tid]) >= 2:
                prev = self.track_history[tid][-2]
                prev_bbox = prev['bbox']
                prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
                prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
                vx = cx - prev_cx
                vy = cy - prev_cy

            # Normalize features
            feat = [
                cx / 640,      # Normalized x
                cy / 480,      # Normalized y
                w / 640,       # Normalized width
                h / 480,       # Normalized height
                vx / 50,       # Normalized velocity x
                vy / 50,       # Normalized velocity y
                np.sqrt(vx**2 + vy**2) / 50,  # Speed
                np.arctan2(vy, vx) / np.pi,   # Direction
            ]
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def _build_adjacency(self, tracks: List[Dict]) -> np.ndarray:
        """Build adjacency matrix based on spatial proximity"""
        n = len(tracks)
        adj = np.zeros((n, n), dtype=np.float32)

        centers = []
        for track in tracks:
            bbox = track['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))

        for i in range(n):
            for j in range(n):
                if i == j:
                    adj[i, j] = 1.0  # Self-connection
                else:
                    dist = np.sqrt(
                        (centers[i][0] - centers[j][0])**2 +
                        (centers[i][1] - centers[j][1])**2
                    )
                    if dist < self.distance_threshold:
                        # Edge weight inversely proportional to distance
                        adj[i, j] = 1.0 - (dist / self.distance_threshold)

        return adj

    def get_interaction_score(self, track_id1: int, track_id2: int,
                              embeddings: Dict[int, np.ndarray]) -> float:
        """
        Compute interaction score between two vehicles
        Higher score = stronger interaction (e.g., yielding behavior)
        """
        if track_id1 not in embeddings or track_id2 not in embeddings:
            return 0.0

        emb1 = embeddings[track_id1]
        emb2 = embeddings[track_id2]

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )

        return float(similarity)

    def detect_yielding_behavior(self,
                                  regular_vehicle_id: int,
                                  emergency_vehicle_id: int,
                                  embeddings: Dict[int, np.ndarray],
                                  tracks: List[Dict]) -> Tuple[bool, float]:
        """
        Detect if a regular vehicle is yielding to an emergency vehicle

        Returns:
            (is_yielding, confidence)
        """
        interaction_score = self.get_interaction_score(
            regular_vehicle_id, emergency_vehicle_id, embeddings
        )

        # Get velocity changes
        is_slowing = False
        if regular_vehicle_id in self.track_history:
            history = list(self.track_history[regular_vehicle_id])
            if len(history) >= 3:
                # Check if vehicle is decelerating
                speeds = []
                for i in range(1, len(history)):
                    prev = history[i-1]['bbox']
                    curr = history[i]['bbox']
                    dx = (curr[0] + curr[2])/2 - (prev[0] + prev[2])/2
                    dy = (curr[1] + curr[3])/2 - (prev[1] + prev[3])/2
                    speeds.append(np.sqrt(dx**2 + dy**2))

                if len(speeds) >= 2 and speeds[-1] < speeds[-2] * 0.8:
                    is_slowing = True

        # Combine signals
        confidence = interaction_score * 0.6
        if is_slowing:
            confidence += 0.4

        is_yielding = confidence > 0.5

        return is_yielding, confidence
