#!/usr/bin/env python3
"""Train LSTM trajectory predictor for collision risk module."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.core import ByteTracker, VehicleDetector
from src.core.collision_risk import TrajectoryPredictor


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


def build_samples_from_video(
    video_path: Path,
    detector: VehicleDetector,
    history_length: int,
    prediction_horizon: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, int]]:
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], {"frames": 0, "tracks": 0, "samples": 0}

    trajectories: Dict[int, List[Tuple[float, float, float, float]]] = {}
    last_center: Dict[int, Tuple[float, float]] = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        detections = detector.detect_vehicles(frame)
        tracks = tracker.update(detections)

        active_ids = set()
        for track in tracks:
            tid = int(track.track_id)
            x1, y1, x2, y2 = track.bbox
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            prev = last_center.get(tid)
            if prev is None:
                vx, vy = 0.0, 0.0
            else:
                vx, vy = cx - prev[0], cy - prev[1]

            trajectories.setdefault(tid, []).append((cx, cy, vx, vy))
            last_center[tid] = (cx, cy)
            active_ids.add(tid)

        stale_ids = [tid for tid in last_center if tid not in active_ids]
        for tid in stale_ids:
            del last_center[tid]

    cap.release()

    input_samples: List[np.ndarray] = []
    target_samples: List[np.ndarray] = []
    min_len = history_length + prediction_horizon
    for points in trajectories.values():
        if len(points) < min_len:
            continue

        arr = np.asarray(points, dtype=np.float32)
        for end in range(history_length, len(arr) - prediction_horizon + 1):
            hist = arr[end - history_length:end]
            fut = arr[end:end + prediction_horizon, :2]
            # Normalize by history mean to improve stability.
            mean_pos = hist[:, :2].mean(axis=0, keepdims=True)
            hist_norm = hist.copy()
            hist_norm[:, :2] -= mean_pos
            fut_norm = fut - mean_pos
            input_samples.append(hist_norm)
            target_samples.append(fut_norm)

    stats = {
        "frames": frame_count,
        "tracks": len(trajectories),
        "samples": len(input_samples),
    }
    return input_samples, target_samples, stats


def split_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    indices = list(range(len(inputs)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)

    split_at = int(len(indices) * (1.0 - val_ratio))
    train_idx = indices[:split_at]
    val_idx = indices[split_at:] or indices[:1]

    return (
        inputs[train_idx],
        targets[train_idx],
    ), (
        inputs[val_idx],
        targets[val_idx],
    )


def create_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_model(
    model: TrajectoryPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[TrajectoryPredictor, List[Dict[str, float]], float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    history: List[Dict[str, float]] = []
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_losses.append(float(F.smooth_l1_loss(pred, y).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch:03d}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_val


def main() -> int:
    parser = argparse.ArgumentParser(description="Train collision trajectory predictor")
    parser.add_argument("--source", type=str, default=None, help="Video file or directory for sample extraction")
    parser.add_argument("--dataset", type=str, default=None, help="Optional prebuilt dataset npz path")
    parser.add_argument("--model", type=str, default="models/yolo12n_vehicle.pt", help="Detector model for extraction")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence for extraction")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Training/extraction device")
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--prediction-horizon", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="experiments", help="Output directory")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = Path(args.output)
    run_name = f"collision_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        data = np.load(args.dataset)
        inputs = data["inputs"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        extraction_stats = {"from_dataset": args.dataset, "samples": int(len(inputs))}
    else:
        if not args.source:
            raise ValueError("--source is required when --dataset is not provided")

        sources = parse_sources(args.source)
        if not sources:
            raise ValueError(f"No videos found from source: {args.source}")

        detector = VehicleDetector(
            model_path=args.model,
            confidence=args.confidence,
            device=args.device,
        )

        all_inputs: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []
        per_video_stats = []
        for video in sources:
            xs, ys, stats = build_samples_from_video(
                video,
                detector,
                history_length=args.history_length,
                prediction_horizon=args.prediction_horizon,
            )
            if xs:
                all_inputs.extend(xs)
                all_targets.extend(ys)
            stats["video"] = str(video)
            per_video_stats.append(stats)
            print(f"Collected {stats['samples']} samples from {video.name}")

        if not all_inputs:
            raise RuntimeError("No training samples collected, try a longer video or lower detection threshold")

        inputs = np.asarray(all_inputs, dtype=np.float32)
        targets = np.asarray(all_targets, dtype=np.float32)
        np.savez_compressed(run_dir / "dataset.npz", inputs=inputs, targets=targets)
        extraction_stats = {"videos": per_video_stats, "samples": int(len(inputs))}

    (x_train, y_train), (x_val, y_val) = split_dataset(inputs, targets, args.val_ratio, args.seed)
    train_loader = create_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = create_loader(x_val, y_val, args.batch_size, shuffle=False)

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = TrajectoryPredictor(
        input_dim=4,
        hidden_dim=64,
        pred_horizon=args.prediction_horizon,
    ).to(device)

    model, loss_history, best_val = train_model(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    checkpoint = {
        "state_dict": model.state_dict(),
        "history_length": args.history_length,
        "prediction_horizon": args.prediction_horizon,
        "best_val_loss": best_val,
    }
    torch.save(checkpoint, run_dir / "best.pt")
    torch.save({**checkpoint, "state_dict": model.state_dict()}, run_dir / "last.pt")

    meta = {
        "run_dir": str(run_dir),
        "device": str(device),
        "dataset_shape": {"inputs": list(inputs.shape), "targets": list(targets.shape)},
        "train_size": int(len(x_train)),
        "val_size": int(len(x_val)),
        "best_val_loss": float(best_val),
        "extraction": extraction_stats,
        "loss_history": loss_history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Training complete. Best val loss: {best_val:.6f}")
    print(f"Saved to: {run_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
