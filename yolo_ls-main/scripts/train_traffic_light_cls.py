#!/usr/bin/env python3
"""
红绿灯状态分类器训练脚本

基于 MobileNetV3-Small 训练 4 分类模型（red / yellow / green / off）。
输入：datasets/traffic_light_cls/{red,yellow,green,off}/ 目录下的 ROI 图片。
输出：models/traffic_light_cls.pt（TorchScript）、models/traffic_light_cls.onnx（ONNX）。

用法:
  # 先采集数据
  python scripts/collect_tl_data.py --source your_video.mp4

  # 训练
  python scripts/train_traffic_light_cls.py --epochs 50 --batch 32

  # 仅评估已有模型
  python scripts/train_traffic_light_cls.py --eval-only --model models/traffic_light_cls.pt
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
# sklearn 替换为纯 numpy 实现，避免依赖问题
def _confusion_matrix(y_true, y_pred):
    """纯 numpy 混淆矩阵"""
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def _classification_report(y_true, y_pred, target_names, zero_division=0):
    """纯 numpy 分类报告"""
    n = len(target_names)
    cm = _confusion_matrix(y_true, y_pred)
    lines = []
    lines.append(f"{'':>16}  precision    recall  f1-score   support")
    lines.append("")
    totals = np.zeros(3)
    total_support = 0
    for i in range(n):
        tp = cm[i, i]
        pred_total = cm[:, i].sum()
        true_total = cm[i, :].sum()
        p = tp / max(pred_total, 1) if pred_total > 0 else 0.0
        r = tp / max(true_total, 1) if true_total > 0 else 0.0
        f1 = 2 * p * r / max(p + r, 1e-9)
        lines.append(f"{target_names[i]:>16}  {p:8.4f}  {r:8.4f}  {f1:8.4f}  {true_total:8d}")
        totals += np.array([p * true_total, r * true_total, f1 * true_total])
        total_support += true_total
    lines.append("")
    if total_support > 0:
        totals /= total_support
    lines.append(f"{'macro avg':>16}  {totals[0]:8.4f}  {totals[1]:8.4f}  {totals[2]:8.4f}  {total_support:8d}")
    return "\n".join(lines)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
CLASS_NAMES = ["green", "off", "red", "yellow"]  # 按字母序，与文件夹名一致
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SIZE = 96  # MobileNetV3 最小输入 224，我们缩放到此尺寸（红绿灯 ROI 通常很小）


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
class TrafficLightDataset(Dataset):
    """从文件夹结构加载红绿灯 ROI 数据集"""

    # 训练时的数据增强（模拟不同光照/角度）
    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    # 验证/测试时的变换（无增强）
    EVAL_TRANSFORM = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, root_dir: str, train: bool = True):
        self.root_dir = Path(root_dir)
        self.train = train
        self.transform = self.TRAIN_TRANSFORM if train else self.EVAL_TRANSFORM
        self.samples: list = []

        for label_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = self.root_dir / cls_name
            if not cls_dir.is_dir():
                print(f"[WARN] Directory not found: {cls_dir} — skipping class '{cls_name}'")
                continue
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append((str(img_path), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {self.root_dir}. "
                f"Run collect_tl_data.py first to gather training data."
            )

        # 类别分布
        self.class_counts = {}
        for _, lbl in self.samples:
            self.class_counts[CLASS_NAMES[lbl]] = self.class_counts.get(CLASS_NAMES[lbl], 0) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # OpenCV 读取以兼容 BGR 摄像头采集的图片
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            # 回退：返回一个黑色图像
            img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # PIL Image
        from PIL import Image
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_model(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """构建 MobileNetV3-Small 分类器（4 类输出）"""
    if pretrained:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small(weights=None)

    # 替换 classifier head
    # MobileNetV3-Small: backbone → classifier[0](Linear 576→1024) → ... → classifier[3](Linear 1024→1000)
    # backbone 实际输出 576 维（adaptive avg pool 后），取 classifier[0] 的 in_features
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, NUM_CLASSES),
    )
    return model


# ---------------------------------------------------------------------------
# 训练 / 评估
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float, list, list]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return running_loss / max(total, 1), correct / max(total, 1), all_preds, all_labels


def export_torchscript(model, save_path: str, device):
    """导出 TorchScript 模型（用于 C++/Python 无 Python 依赖推理）"""
    model.eval()
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced.save(save_path)
    print(f"[OK] TorchScript exported to {save_path}")

    # 验证导出
    loaded = torch.jit.load(save_path)
    with torch.no_grad():
        out1 = model(example)
        out2 = loaded(example)
    assert torch.allclose(out1, out2, atol=1e-5), "TorchScript export verification failed!"
    print("[OK] TorchScript export verified (outputs match)")


def export_onnx(model, save_path: str, device):
    """导出 ONNX 模型"""
    try:
        model.eval()
        example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
        torch.onnx.export(
            model, example, save_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13,
        )
        print(f"[OK] ONNX exported to {save_path}")
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="红绿灯状态分类器训练")
    parser.add_argument("--data", default="datasets/traffic_light_cls",
                        help="数据集根目录")
    parser.add_argument("--output", default="models/traffic_light_cls.pt",
                        help="输出模型路径")
    parser.add_argument("--output-onnx", default="models/traffic_light_cls.onnx",
                        help="ONNX 输出路径")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout 比例")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="不使用 ImageNet 预训练权重")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已有模型")
    parser.add_argument("--model", default=None,
                        help="评估时使用的模型路径")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 仅评估模式 ----
    if args.eval_only:
        model_path = args.model or args.output
        if not Path(model_path).exists():
            print(f"[ERROR] Model not found: {model_path}")
            return 1
        model = build_model(pretrained=False)
        loaded = torch.jit.load(model_path, map_location=device)
        # TorchScript 模型直接评估
        full_dataset = TrafficLightDataset(args.data, train=False)
        loader = DataLoader(full_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
        # 对于 TorchScript，用原生方式评估
        all_preds, all_labels = [], []
        loaded.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = loaded(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.tolist())
        print("\nClassification Report (on full dataset):")
        print(_classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES,
            zero_division=0,
        ))
        print("Confusion Matrix:")
        print(_confusion_matrix(all_labels, all_preds))
        return 0

    # ---- 训练模式 ----
    full_dataset = TrafficLightDataset(args.data, train=True)
    print(f"\nDataset loaded: {len(full_dataset)} samples")
    print("Class distribution:")
    for cls_name, count in full_dataset.class_counts.items():
        print(f"  {cls_name}: {count}")

    # 划分训练/验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # 构建模型
    model = build_model(pretrained=not args.no_pretrained, dropout=args.dropout)
    model = model.to(device)

    # 类别权重（处理不均衡）
    class_counts = torch.tensor(
        [full_dataset.class_counts.get(c, 1) for c in CLASS_NAMES],
        dtype=torch.float32,
    )
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device,
        )
        scheduler.step()

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            marker = " <-- BEST"
            # 保存最佳模型
            best_path = Path(args.output).with_suffix(".best.pt")
            export_torchscript(model, str(best_path), device)

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{marker}")

    print("-" * 60)
    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

    # 最终评估
    print("\nFinal Validation Report:")
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    print(_classification_report(
        val_labels, val_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(_confusion_matrix(val_labels, val_preds))

    # 导出最终模型
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_torchscript(model, str(output_path), device)

    # 导出 ONNX
    onnx_path = args.output_onnx
    if onnx_path:
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        export_onnx(model, onnx_path, device)

    # 保存训练历史
    history_path = output_path.with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # 保存类别映射
    class_map = {i: name for i, name in enumerate(CLASS_NAMES)}
    class_map_path = output_path.parent / "traffic_light_cls_classes.json"
    with open(class_map_path, "w") as f:
        json.dump(class_map, f, indent=2)

    print(f"\nDone! Model saved to {output_path}")
    print(f"Class map saved to {class_map_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
