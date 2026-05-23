"""
YOLO模型训练脚本
支持多种模型大小的训练和对比；训练完成后可复制到 config 中的 models/yolo12n_vehicle.pt
"""
import argparse
import shutil
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from src.utils.config import load_config
except ImportError:
    load_config = None


def train_model(
    model_name: str = "yolo12n.pt",
    data_yaml: str = "datasets/traffic.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0",
    project: str = "experiments",
    name: str = None,
    resume: bool = False,
    pretrained: bool = True,
):
    """
    训练YOLO模型

    Args:
        model_name: 模型名称 (yolo12n.pt, yolo12s.pt, yolo12m.pt等)
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图片尺寸
        device: 训练设备 (0, 1, cpu)
        project: 实验保存目录
        name: 实验名称
        resume: 是否从上次中断处继续训练
        pretrained: 是否使用预训练权重
    """
    # 生成实验名称
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_base = Path(model_name).stem
        name = f"{model_base}_{timestamp}"

    print("=" * 60)
    print("YOLO 模型训练")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"数据集: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图片尺寸: {img_size}")
    print(f"设备: {device}")
    print(f"实验名称: {name}")
    print("=" * 60)

    # 加载模型
    if pretrained:
        model = YOLO(model_name)
    else:
        # 从头训练（不推荐）
        model = YOLO(model_name.replace(".pt", ".yaml"))

    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        resume=resume,
        # 训练参数
        patience=50,          # 早停耐心值
        save=True,            # 保存检查点
        save_period=10,       # 每10轮保存一次
        val=True,             # 训练时验证
        plots=True,           # 生成训练曲线图
        # 优化器参数
        optimizer="auto",     # 自动选择优化器
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率因子
        momentum=0.937,       # SGD动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3.0,    # 预热轮数
        warmup_momentum=0.8,  # 预热动量
        # 数据增强
        hsv_h=0.015,          # 色调增强
        hsv_s=0.7,            # 饱和度增强
        hsv_v=0.4,            # 明度增强
        degrees=0.0,          # 旋转角度
        translate=0.1,        # 平移
        scale=0.5,            # 缩放
        fliplr=0.5,           # 水平翻转
        mosaic=1.0,           # 马赛克增强
        mixup=0.0,            # MixUp增强
        # 其他
        verbose=True,
        seed=42,
    )

    print("\n训练完成!")
    print(f"最佳模型保存在: {project}/{name}/weights/best.pt")
    print(f"最后模型保存在: {project}/{name}/weights/last.pt")

    return results, f"{project}/{name}/weights/best.pt"


def main():
    parser = argparse.ArgumentParser(description="YOLO模型训练")
    parser.add_argument("--model", "-m", default="yolo12n.pt",
                        help="模型名称 (yolo12n.pt, yolo12s.pt, yolo12m.pt)")
    parser.add_argument("--data", "-d", default="datasets/traffic.yaml",
                        help="数据集配置文件")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch", "-b", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--img-size", type=int, default=640,
                        help="输入图片尺寸")
    parser.add_argument("--device", default="0",
                        help="训练设备 (0, 1, cpu)")
    parser.add_argument("--project", default="experiments",
                        help="实验保存目录")
    parser.add_argument("--name", default=None,
                        help="实验名称")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处继续训练")
    parser.add_argument("--config", default="config/settings.yaml",
                        help="读取 detector.model_path 作为默认输出路径")
    parser.add_argument("--output-model", default=None,
                        help="训练后将 best.pt 复制到此路径（默认读 config）")
    args = parser.parse_args()

    output_model = args.output_model
    if output_model is None and load_config is not None:
        cfg = load_config(args.config)
        output_model = cfg.get('detector', {}).get('model_path', 'models/yolo12n_vehicle.pt')

    _, best_path = train_model(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    if output_model:
        src = Path(best_path)
        if not src.is_absolute():
            src = ROOT / src
        dst = ROOT / output_model
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Deployed model to {dst} (update config detector.model_path if needed)")
        else:
            print(f"Warning: best weights not found at {src}")


if __name__ == "__main__":
    main()
