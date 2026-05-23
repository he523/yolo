#!/usr/bin/env python3
"""Generate baseline evaluation metrics and report for graduation project."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core import AdaptiveViolationDetector, ByteTracker, FeatureExtractor, VehicleDetector
from src.ocr.plate_reader import PlateOCR


def find_latest_results_csv() -> Optional[Path]:
    candidates = list((ROOT / "runs" / "detect").glob("**/results.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_detection_training(csv_path: Path) -> Dict:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})

    if not rows:
        return {"error": "results.csv is empty", "path": str(csv_path)}

    def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default))
        except (TypeError, ValueError):
            return default

    last = rows[-1]
    best_map50_row = max(rows, key=lambda r: as_float(r, "metrics/mAP50(B)"))
    best_map95_row = max(rows, key=lambda r: as_float(r, "metrics/mAP50-95(B)"))

    return {
        "path": str(csv_path),
        "epochs": int(as_float(last, "epoch", 0)),
        "final": {
            "precision": as_float(last, "metrics/precision(B)"),
            "recall": as_float(last, "metrics/recall(B)"),
            "map50": as_float(last, "metrics/mAP50(B)"),
            "map50_95": as_float(last, "metrics/mAP50-95(B)"),
            "train_box_loss": as_float(last, "train/box_loss"),
            "train_cls_loss": as_float(last, "train/cls_loss"),
        },
        "best": {
            "map50": as_float(best_map50_row, "metrics/mAP50(B)"),
            "map50_epoch": int(as_float(best_map50_row, "epoch", 0)),
            "map50_95": as_float(best_map95_row, "metrics/mAP50-95(B)"),
            "map50_95_epoch": int(as_float(best_map95_row, "epoch", 0)),
        },
    }


def load_ocr_val_samples(val_file: Path, sample_size: int, seed: int) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    with val_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            samples.append((parts[0], parts[1]))

    rnd = random.Random(seed)
    if len(samples) > sample_size:
        samples = rnd.sample(samples, sample_size)

    return samples


def evaluate_ocr(
    model_path: Path,
    val_file: Path,
    image_root: Path,
    sample_size: int,
    seed: int,
    use_gpu: bool,
) -> Dict:
    if not model_path.exists():
        return {"error": f"OCR model not found: {model_path}"}
    if not val_file.exists():
        return {"error": f"OCR val file not found: {val_file}"}

    ocr = PlateOCR(model_path=str(model_path), use_gpu=use_gpu)
    if ocr.model is None:
        return {"error": "Failed to load OCR model"}

    samples = load_ocr_val_samples(val_file, sample_size, seed)
    if not samples:
        return {"error": "No OCR validation samples"}

    total = 0
    matched = 0
    invalid_img = 0
    predicted_count = 0
    confidences: List[float] = []

    started = time.perf_counter()
    for rel_path, truth in samples:
        img_path = image_root / rel_path
        img = cv2.imread(str(img_path))
        if img is None:
            invalid_img += 1
            continue

        h, w = img.shape[:2]
        result = ocr.recognize(img, (0, 0, w, h))

        total += 1
        if result:
            predicted_count += 1
            confidences.append(float(result.confidence))
            if result.plate_number == truth:
                matched += 1

    elapsed = time.perf_counter() - started

    return {
        "model_path": str(model_path),
        "val_file": str(val_file),
        "sample_size": len(samples),
        "valid_images": total,
        "invalid_images": invalid_img,
        "predicted_count": predicted_count,
        "exact_match_count": matched,
        "exact_match_accuracy": matched / max(1, total),
        "prediction_coverage": predicted_count / max(1, total),
        "avg_confidence": statistics.mean(confidences) if confidences else 0.0,
        "eval_seconds": elapsed,
    }


def benchmark_runtime(
    detector_model: Path,
    image_dir: Path,
    sample_size: int,
    warmup: int,
    confidence: float,
    device: str,
    seed: int,
) -> Dict:
    if not detector_model.exists():
        return {"error": f"Detector model not found: {detector_model}"}
    if not image_dir.exists():
        return {"error": f"Image directory not found: {image_dir}"}

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        return {"error": f"No images under {image_dir}"}

    if len(image_paths) > sample_size:
        image_paths = image_paths[:sample_size]

    detector = VehicleDetector(model_path=str(detector_model), confidence=confidence, device=device)
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
    feature_extractor = FeatureExtractor(pixel_to_meter=0.05, fps=15)
    violation_detector = AdaptiveViolationDetector(speed_limit=60, snapshot_dir="/tmp/yolo_ls_eval_snapshots")

    def maybe_sync_cuda() -> None:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    loaded_frames = []
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is not None:
            loaded_frames.append(frame)

    if not loaded_frames:
        return {"error": "No valid images loaded for runtime benchmark"}

    for frame in loaded_frames[:warmup]:
        detections = detector.detect_vehicles(frame)
        tracks = tracker.update(detections)
        vehicle_bboxes = [t.bbox for t in tracks]
        violation_detector.update(frame, vehicle_bboxes)
        for track in tracks:
            feature_extractor.extract(frame, track.track_id, track.bbox)

    frame_times: List[float] = []
    det_counts: List[int] = []

    maybe_sync_cuda()
    for frame in loaded_frames:
        started = time.perf_counter()
        detections = detector.detect_vehicles(frame)
        tracks = tracker.update(detections)
        vehicle_bboxes = [t.bbox for t in tracks]
        violation_detector.update(frame, vehicle_bboxes)
        for track in tracks:
            feature_extractor.extract(frame, track.track_id, track.bbox)
        maybe_sync_cuda()
        ended = time.perf_counter()

        frame_times.append(ended - started)
        det_counts.append(len(detections))

    avg_ms = statistics.mean(frame_times) * 1000
    fps = 1.0 / statistics.mean(frame_times)

    return {
        "device": device,
        "model": str(detector_model),
        "num_frames": len(loaded_frames),
        "avg_latency_ms": avg_ms,
        "p95_latency_ms": float(np.percentile(frame_times, 95) * 1000),
        "fps": fps,
        "avg_detections_per_frame": statistics.mean(det_counts) if det_counts else 0.0,
    }


def run_pytest() -> Dict:
    started = time.perf_counter()
    proc = subprocess.run(
        ["pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    summary_match = re.search(r"(\d+) passed(?:,\s*(\d+) warnings?)?", output)

    result = {
        "return_code": proc.returncode,
        "duration_seconds": elapsed,
        "raw_summary": output.strip().splitlines()[-1] if output.strip() else "",
    }

    if summary_match:
        result["passed"] = int(summary_match.group(1))
        result["warnings"] = int(summary_match.group(2) or 0)

    return result


def generate_markdown(report: Dict) -> str:
    timestamp = report["generated_at"]
    det = report.get("detection_training", {})
    ocr = report.get("ocr_eval", {})
    runtime = report.get("runtime_benchmark", {})
    tests = report.get("pytest", {})

    lines = [
        "# Baseline Evaluation Report",
        "",
        f"- Generated at: {timestamp}",
        f"- Repo: {report.get('repo')}",
        "",
        "## 1) Detection Training Metrics",
    ]

    if "error" in det:
        lines.append(f"- Error: {det['error']}")
    else:
        lines.extend(
            [
                f"- Source: `{det['path']}`",
                f"- Final epoch: {det['epochs']}",
                f"- Final Precision: {det['final']['precision']:.4f}",
                f"- Final Recall: {det['final']['recall']:.4f}",
                f"- Final mAP@50: {det['final']['map50']:.4f}",
                f"- Final mAP@50-95: {det['final']['map50_95']:.4f}",
                f"- Best mAP@50: {det['best']['map50']:.4f} (epoch {det['best']['map50_epoch']})",
                f"- Best mAP@50-95: {det['best']['map50_95']:.4f} (epoch {det['best']['map50_95_epoch']})",
            ]
        )

    lines.extend(["", "## 2) OCR Sample Evaluation"])
    if "error" in ocr:
        lines.append(f"- Error: {ocr['error']}")
    else:
        lines.extend(
            [
                f"- Model: `{ocr['model_path']}`",
                f"- Validation file: `{ocr['val_file']}`",
                f"- Sample size: {ocr['sample_size']}",
                f"- Valid images: {ocr['valid_images']}",
                f"- Exact-match accuracy: {ocr['exact_match_accuracy']:.4f}",
                f"- Prediction coverage: {ocr['prediction_coverage']:.4f}",
                f"- Average confidence (predicted): {ocr['avg_confidence']:.4f}",
                f"- Eval time: {ocr['eval_seconds']:.2f}s",
            ]
        )

    lines.extend(["", "## 3) Runtime Benchmark"])
    if "error" in runtime:
        lines.append(f"- Error: {runtime['error']}")
    else:
        lines.extend(
            [
                f"- Model: `{runtime['model']}`",
                f"- Device: {runtime['device']}",
                f"- Frames: {runtime['num_frames']}",
                f"- Avg latency: {runtime['avg_latency_ms']:.2f} ms",
                f"- P95 latency: {runtime['p95_latency_ms']:.2f} ms",
                f"- Throughput: {runtime['fps']:.2f} FPS",
                f"- Avg detections/frame: {runtime['avg_detections_per_frame']:.2f}",
            ]
        )

    lines.extend(["", "## 4) Test Status"])
    lines.extend(
        [
            f"- Return code: {tests.get('return_code')}",
            f"- Passed: {tests.get('passed', 'N/A')}",
            f"- Warnings: {tests.get('warnings', 'N/A')}",
            f"- Duration: {tests.get('duration_seconds', 0):.2f}s",
            f"- Summary: {tests.get('raw_summary', '')}",
        ]
    )

    lines.extend(
        [
            "",
            "## 5) Conclusion",
            "- The project has a runnable end-to-end baseline suitable for thesis demonstrations.",
            "- Next step for stronger academic results: add controlled ablation and scenario-based error analysis.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate baseline evaluation report")
    parser.add_argument("--ocr-samples", type=int, default=200, help="Number of OCR validation samples")
    parser.add_argument("--bench-samples", type=int, default=120, help="Number of images for runtime benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for runtime benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Benchmark device")
    args = parser.parse_args()

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    latest_csv = find_latest_results_csv()
    detection_training = (
        parse_detection_training(latest_csv)
        if latest_csv is not None
        else {"error": "No YOLO training results.csv found under runs/detect"}
    )

    ocr_eval = evaluate_ocr(
        model_path=ROOT / "models" / "plate_ocr.pt",
        val_file=ROOT / "datasets" / "cblprd" / "val.txt",
        image_root=ROOT / "datasets" / "cblprd",
        sample_size=args.ocr_samples,
        seed=args.seed,
        use_gpu=args.device == "cuda",
    )

    runtime_benchmark = benchmark_runtime(
        detector_model=ROOT / "models" / "yolo12n_vehicle.pt",
        image_dir=ROOT / "datasets" / "vehicle_detection" / "valid" / "images",
        sample_size=args.bench_samples,
        warmup=args.warmup,
        confidence=0.2,
        device=args.device,
        seed=args.seed,
    )

    pytest_result = run_pytest()

    report = {
        "generated_at": generated_at,
        "repo": "https://github.com/Zhye26/yolo_ls.git",
        "detection_training": detection_training,
        "ocr_eval": ocr_eval,
        "runtime_benchmark": runtime_benchmark,
        "pytest": pytest_result,
    }

    output_dir = ROOT / "docs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "baseline_metrics.json"
    md_path = output_dir / "baseline_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(generate_markdown(report), encoding="utf-8")

    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved report Markdown: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
