"""
车牌识别模型训练脚本
基于 CRNN (CNN + RNN + CTC) 架构
"""
import argparse
import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# 中国车牌字符集
CHARS = [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '警', '学', '挂',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '-'  # blank for CTC
]
CHAR2IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX2CHAR = {idx: char for idx, char in enumerate(CHARS)}
NUM_CLASSES = len(CHARS)
BLANK_IDX = CHAR2IDX['-']


class PlateDataset(Dataset):
    """车牌数据集"""

    def __init__(self, data_file: str, img_dir: str, transform=None, max_samples: int = None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples = []

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_path = parts[0]
                    plate_number = parts[1]
                    # 过滤无效字符
                    if all(c in CHAR2IDX or c == '·' for c in plate_number.replace('·', '')):
                        self.samples.append((img_path, plate_number.replace('·', '')))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, plate_number = self.samples[idx]
        img_full_path = self.img_dir.parent / img_path

        try:
            img = Image.open(img_full_path).convert('RGB')
        except Exception:
            # 返回空白图像
            img = Image.new('RGB', (94, 24), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        # 编码标签
        label = [CHAR2IDX.get(c, BLANK_IDX) for c in plate_number]
        label_length = len(label)

        return img, torch.tensor(label, dtype=torch.long), label_length


def collate_fn(batch):
    """自定义 collate 函数处理变长标签"""
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images, 0)

    # 填充标签到相同长度
    max_len = max(label_lengths)
    padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    return images, padded_labels, torch.tensor(label_lengths, dtype=torch.long)


class CRNN(nn.Module):
    """CRNN 车牌识别模型"""

    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()

        # CNN 特征提取 (输入: 32x128)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),  # 16x64
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),  # 8x32
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),  # 4x32
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),  # 2x32
            nn.Conv2d(512, 512, (2, 1), 1, 0), nn.ReLU(),  # 1x32
        )

        # RNN 序列建模
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)

        # 分类器
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # CNN: (B, 3, 32, 128) -> (B, 512, 1, 32)
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2).permute(0, 2, 1)  # (B, W, C)

        # RNN
        rnn_out, _ = self.rnn(conv)

        # 分类
        output = self.fc(rnn_out)  # (B, W, num_classes)
        return output.permute(1, 0, 2)  # (W, B, num_classes) for CTC


def decode_predictions(preds, idx2char):
    """CTC 解码"""
    preds = preds.argmax(2).permute(1, 0).cpu().numpy()
    results = []
    for pred in preds:
        chars = []
        prev = -1
        for p in pred:
            if p != prev and p != BLANK_IDX:
                chars.append(idx2char.get(p, ''))
            prev = p
        results.append(''.join(chars))
    return results


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels, label_lengths in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)
        loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = decode_predictions(outputs, IDX2CHAR)

            for i, pred in enumerate(preds):
                label_len = label_lengths[i].item()
                label = ''.join([IDX2CHAR.get(l.item(), '') for l in labels[i, :label_len]])
                if pred == label:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 数据集
    train_dataset = PlateDataset(args.train_file, args.img_dir, train_transform, args.max_samples)
    val_dataset = PlateDataset(args.val_file, args.img_dir, val_transform, args.max_samples // 5 if args.max_samples else None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

    # 模型
    model = CRNN(NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.output) / f"plate_ocr_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("车牌识别模型训练")
    print("=" * 60)
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"保存目录: {save_dir}")
    print("=" * 60)

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"  -> 保存最佳模型 (Acc: {best_acc:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_dir / f"epoch{epoch+1}.pt")

    torch.save(model.state_dict(), save_dir / "last.pt")
    print(f"\n训练完成! 最佳准确率: {best_acc:.4f}")
    print(f"模型保存在: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="车牌识别模型训练")
    parser.add_argument("--train-file", default="datasets/cblprd/train.txt", help="训练集标注文件")
    parser.add_argument("--val-file", default="datasets/cblprd/val.txt", help="验证集标注文件")
    parser.add_argument("--img-dir", default="datasets/cblprd/CBLPRD-330k", help="图像目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--device", default="0", help="设备")
    parser.add_argument("--output", default="experiments", help="输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="最大样本数（用于快速测试）")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
