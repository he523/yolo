#!/usr/bin/env python3
"""Train ST-GAT interaction encoder with adjacency reconstruction objective."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.core import ByteTracker, VehicleDetector
from src.core.stgat import SpatioTemporalGAT


def parse_sources(source: str) -> List[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        videos: List[Path] = []
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            videos.extend(sorted(path.glob(ext)))
        return videos
    return []


def build_adjacency(centers: List[Tuple[float, float]], distance_threshold: float) -> np.ndarray:
    n = len(centers)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                adj[i, j] = 1.0
                continue
            dist = np.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
            if dist < distance_threshold:
                adj[i, j] = 1.0 - (dist / distance_threshold)
    return adj


def extract_graph_snapshots(
    video: Path,
    detector: VehicleDetector,
    distance_threshold: float,
    max_snapshots: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict[str, int]]:
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [], {"frames": 0, "snapshots": 0}

    snapshots: List[Tuple[np.ndarray, np.ndarray]] = []
    prev_centers: Dict[int, Tuple[float, float]] = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        detections = detector.detect_vehicles(frame)
        tracks = tracker.update(detections)
        if len(tracks) < 2:
            continue

        features = []
        centers = []
        active_ids = set()
        for track in tracks:
            tid = int(track.track_id)
            x1, y1, x2, y2 = track.bbox
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            w = float(x2 - x1)
            h = float(y2 - y1)
            prev = prev_centers.get(tid)
            if prev is None:
                vx, vy = 0.0, 0.0
            else:
                vx, vy = cx - prev[0], cy - prev[1]
            speed = np.sqrt(vx ** 2 + vy ** 2)
            direction = np.arctan2(vy, vx) / np.pi

            features.append([
                cx / 640.0,
                cy / 480.0,
                w / 640.0,
                h / 480.0,
                vx / 50.0,
                vy / 50.0,
                speed / 50.0,
                direction,
            ])
            centers.append((cx, cy))
            prev_centers[tid] = (cx, cy)
            active_ids.add(tid)

        stale = [tid for tid in prev_centers if tid not in active_ids]
        for tid in stale:
            del prev_centers[tid]

        features_arr = np.asarray(features, dtype=np.float32)
        adj_arr = build_adjacency(centers, distance_threshold)
        snapshots.append((features_arr, adj_arr))

        if max_snapshots > 0 and len(snapshots) >= max_snapshots:
            break

    cap.release()
    return snapshots, {"frames": frame_count, "snapshots": len(snapshots)}


def adjacency_reconstruction_loss(embeddings: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    emb_norm = F.normalize(embeddings, dim=1)
    similarity = torch.matmul(emb_norm, emb_norm.transpose(0, 1))
    similarity = (similarity + 1.0) / 2.0
    return F.mse_loss(similarity, adj)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ST-GAT interaction encoder")
    parser.add_argument("--source", type=str, required=True, help="Video file or directory")
    parser.add_argument("--model", type=str, default="models/yolo12n_vehicle.pt", help="Detector model for extraction")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--distance-threshold", type=float, default=200.0, help="Graph edge distance threshold")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-snapshots", type=int, default=4000, help="Max graph snapshots for training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", type=str, default="experiments", help="Output directory")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sources = parse_sources(args.source)
    if not sources:
        raise ValueError(f"No videos found from source: {args.source}")

    detector = VehicleDetector(
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
    )

    snapshots: List[Tuple[np.ndarray, np.ndarray]] = []
    per_video_stats = []
    per_video_limit = max(args.max_snapshots // max(len(sources), 1), 1)

    for video in sources:
        data, stats = extract_graph_snapshots(
            video,
            detector,
            distance_threshold=args.distance_threshold,
            max_snapshots=per_video_limit,
        )
        snapshots.extend(data)
        stats["video"] = str(video)
        per_video_stats.append(stats)
        print(f"Collected {stats['snapshots']} graph snapshots from {video.name}")

    if not snapshots:
        raise RuntimeError("No graph snapshots collected for training")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = SpatioTemporalGAT(node_features=8, hidden_dim=32, output_dim=16, num_heads=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []

    for epoch in range(1, args.epochs + 1):
        random.shuffle(snapshots)
        model.train()
        losses: List[float] = []

        for node_features, adj in snapshots:
            x = torch.from_numpy(node_features).to(device)
            a = torch.from_numpy(adj).to(device)
            emb = model(x, a)
            loss = adjacency_reconstruction_loss(emb, a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"Epoch {epoch:03d}/{args.epochs}: loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    run_name = f"stgat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.output) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "distance_threshold": args.distance_threshold,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, run_dir / "best.pt")
    torch.save(checkpoint, run_dir / "last.pt")

    metrics = {
        "run_dir": str(run_dir),
        "device": str(device),
        "best_loss": float(best_loss),
        "num_snapshots": len(snapshots),
        "videos": per_video_stats,
        "loss_history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Saved to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
