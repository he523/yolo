"""
实验结果可视化脚本
生成训练曲线、对比图表等
"""
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(experiment_dir: str, output_dir: str = None):
    """绘制训练曲线"""
    exp_path = Path(experiment_dir)
    if output_dir is None:
        output_dir = exp_path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取训练结果
    results_csv = exp_path / "results.csv"
    if not results_csv.exists():
        print(f"找不到训练结果: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 损失曲线
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss')
    if 'train/cls_loss' in df.columns:
        ax.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss')
    if 'train/dfl_loss' in df.columns:
        ax.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)

    # 2. mAP曲线
    ax = axes[0, 1]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP')
    ax.set_title('Validation mAP')
    ax.legend()
    ax.grid(True)

    # 3. Precision/Recall曲线
    ax = axes[1, 0]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.set_title('Precision & Recall')
    ax.legend()
    ax.grid(True)

    # 4. 学习率曲线
    ax = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax.plot(df['epoch'], df['lr/pg0'], label='LR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    save_path = output_path / "training_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线保存到: {save_path}")


def plot_comparison_results(comparison_file: str, output_dir: str = None):
    """绘制模型对比图表"""
    with open(comparison_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = Path(comparison_file).parent

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    models = list(results["models"].keys())
    model_names = [m.replace(".pt", "") for m in models]

    # 提取数据
    params = []
    fps = []
    map50 = []
    map50_95 = []

    for model in models:
        data = results["models"][model]
        params.append(data.get("info", {}).get("total_params", 0) / 1e6)
        fps.append(data.get("speed", {}).get("fps", 0))
        map50.append(data.get("accuracy", {}).get("mAP50", 0) * 100)
        map50_95.append(data.get("accuracy", {}).get("mAP50-95", 0) * 100)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 参数量对比
    ax = axes[0, 0]
    bars = ax.bar(model_names, params, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)])
    ax.set_ylabel('Parameters (M)')
    ax.set_title('Model Parameters')
    for bar, val in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}M', ha='center', va='bottom')

    # 2. FPS对比
    ax = axes[0, 1]
    bars = ax.bar(model_names, fps, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)])
    ax.set_ylabel('FPS')
    ax.set_title('Inference Speed (FPS)')
    for bar, val in zip(bars, fps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom')

    # 3. mAP对比
    ax = axes[1, 0]
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, map50, width, label='mAP@50', color='#3498db')
    bars2 = ax.bar(x + width/2, map50_95, width, label='mAP@50-95', color='#2ecc71')
    ax.set_ylabel('mAP (%)')
    ax.set_title('Detection Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()

    # 4. 速度-精度权衡
    ax = axes[1, 1]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)]
    sizes = [p * 20 for p in params]  # 点大小与参数量成正比
    scatter = ax.scatter(fps, map50_95, c=colors, s=sizes, alpha=0.7)
    for i, name in enumerate(model_names):
        ax.annotate(name, (fps[i], map50_95[i]), textcoords="offset points",
                    xytext=(5, 5), ha='left')
    ax.set_xlabel('FPS')
    ax.set_ylabel('mAP@50-95 (%)')
    ax.set_title('Speed-Accuracy Trade-off')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_path / "comparison_charts.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"对比图表保存到: {save_path}")


def generate_report(comparison_file: str, output_dir: str = None):
    """生成实验报告（Markdown格式）"""
    with open(comparison_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = Path(comparison_file).parent

    output_path = Path(output_dir)

    report = []
    report.append("# YOLO模型对比实验报告\n")
    report.append(f"**实验时间**: {results['timestamp']}\n")
    report.append(f"**测试设备**: {results['device']}\n")
    report.append(f"**数据集**: {results['data_yaml']}\n")
    report.append("\n## 实验结果\n")

    # 表格
    report.append("| 模型 | 参数量 | 大小(MB) | FPS | mAP@50 | mAP@50-95 | Precision | Recall |")
    report.append("|------|--------|----------|-----|--------|-----------|-----------|--------|")

    for model, data in results["models"].items():
        params = data.get("info", {}).get("total_params", 0)
        size = data.get("info", {}).get("model_size_mb", 0)
        fps = data.get("speed", {}).get("fps", 0)
        map50 = data.get("accuracy", {}).get("mAP50", 0)
        map50_95 = data.get("accuracy", {}).get("mAP50-95", 0)
        precision = data.get("accuracy", {}).get("precision", 0)
        recall = data.get("accuracy", {}).get("recall", 0)

        report.append(
            f"| {model} | {params/1e6:.2f}M | {size:.1f} | {fps:.1f} | "
            f"{map50:.4f} | {map50_95:.4f} | {precision:.4f} | {recall:.4f} |"
        )

    report.append("\n## 结论\n")
    report.append("根据实验结果，可以得出以下结论：\n")
    report.append("1. **模型大小与精度的权衡**: 更大的模型通常具有更高的精度，但推理速度更慢。\n")
    report.append("2. **实时性要求**: 如果需要实时处理（>30 FPS），建议使用yolo12n或yolo12s。\n")
    report.append("3. **精度优先**: 如果精度是首要考虑因素，建议使用yolo12m或更大的模型。\n")

    # 保存报告
    report_path = output_path / "experiment_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"实验报告保存到: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="实验结果可视化")
    parser.add_argument("--training", "-t", help="训练实验目录")
    parser.add_argument("--comparison", "-c", help="对比实验结果JSON文件")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--report", action="store_true", help="生成实验报告")
    args = parser.parse_args()

    if args.training:
        plot_training_curves(args.training, args.output)

    if args.comparison:
        plot_comparison_results(args.comparison, args.output)
        if args.report:
            generate_report(args.comparison, args.output)


if __name__ == "__main__":
    main()
