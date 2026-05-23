#!/usr/bin/env python3
"""下载/准备默认模型权重（YOLO 预训练 + 创建 models 目录）。"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.model_paths import PROJECT_ROOT, YOLO_FALLBACK_MODELS


def main():
    parser = argparse.ArgumentParser(description="Prepare default model files")
    parser.add_argument(
        "--output-vehicle",
        default="models/yolo12n_vehicle.pt",
        help="Copy/download vehicle detector to this path",
    )
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_path = PROJECT_ROOT / args.output_vehicle
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return 0

    for name in YOLO_FALLBACK_MODELS:
        try:
            from ultralytics import YOLO
            print(f"Downloading {name} via Ultralytics...")
            model = YOLO(name)
            src = Path(model.ckpt_path) if hasattr(model, 'ckpt_path') else PROJECT_ROOT / name
            if not src.exists():
                src = PROJECT_ROOT / name
            if src.exists():
                shutil.copy2(src, out_path)
                print(f"Saved vehicle model to {out_path}")
                return 0
        except Exception as exc:
            print(f"  {name}: {exc}")

    print(
        "Could not auto-download. Manually place weights at:\n"
        f"  {out_path}\n"
        "Or train: python scripts/train.py --output-model models/yolo12n_vehicle.pt"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
