"""
模型对比实验脚本
对比不同YOLO模型的性能（精度、速度、模型大小）
"""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import torch


# 要对比的模型列表
MODELS = [
    "yolo12n.pt",  # Nano - 最小最快
    "yolo12s.pt",  # Small
    "yolo12m.pt",  # Medium
    # "yolo12l.pt",  # Large (可选，需要更多显存)
]


def get_model_info(model_path: str) -> dict:
    """获取模型信息"""
    model = YOLO(model_path)

    # 获取参数量
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)

    # 获取模型大小
    model_size = Path(model_path).stat().st_size / (1024 * 1024) if Path(model_path).exists() else 0

    return {
        "model": model_path,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size,
    }


def benchmark_speed(model_path: str, img_size: int = 640, device: str = "cpu",
                    warmup: int = 10, runs: int = 100) -> dict:
    """测试模型推理速度"""
    model = YOLO(model_path)

    # 创建随机输入
    dummy_input = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

    # 预热
    print(f"  预热 {warmup} 次...")
    for _ in range(warmup):
        model(dummy_input, verbose=False)

    # 正式测试
    print(f"  测试 {runs} 次...")
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        model(dummy_input, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # 转换为毫秒

    times = np.array(times)
    return {
        "device": device,
        "img_size": img_size,
        "runs": runs,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000 / np.mean(times)),
    }


def evaluate_accuracy(model_path: str, data_yaml: str, device: str = "cpu") -> dict:
    """评估模型精度"""
    model = YOLO(model_path)

    # 运行验证
    results = model.val(
        data=data_yaml,
        device=device,
        verbose=False,
        plots=False,
    )

    return {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }


def run_comparison(
    models: list = None,
    data_yaml: str = "datasets/traffic.yaml",
    device: str = "cpu",
    output_dir: str = "experiments/comparison",
):
    """运行模型对比实验"""
    if models is None:
        models = MODELS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "data_yaml": data_yaml,
        "models": {}
    }

    print("=" * 70)
    print("YOLO 模型对比实验")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"数据集: {data_yaml}")
    print(f"对比模型: {models}")
    print("=" * 70)

    for model_path in models:
        print(f"\n{'='*50}")
        print(f"测试模型: {model_path}")
        print("=" * 50)

        model_results = {}

        # 1. 获取模型信息
        print("\n[1/3] 获取模型信息...")
        try:
            info = get_model_info(model_path)
            model_results["info"] = info
            print(f"  参数量: {info['total_params']:,}")
            print(f"  模型大小: {info['model_size_mb']:.2f} MB")
        except Exception as e:
            print(f"  错误: {e}")
            model_results["info"] = {"error": str(e)}

        # 2. 测试推理速度
        print("\n[2/3] 测试推理速度...")
        try:
            speed = benchmark_speed(model_path, device=device, warmup=5, runs=50)
            model_results["speed"] = speed
            print(f"  平均耗时: {speed['mean_ms']:.2f} ms")
            print(f"  FPS: {speed['fps']:.1f}")
        except Exception as e:
            print(f"  错误: {e}")
            model_results["speed"] = {"error": str(e)}

        # 3. 评估精度（如果有数据集）
        print("\n[3/3] 评估精度...")
        try:
            if Path(data_yaml).exists():
                accuracy = evaluate_accuracy(model_path, data_yaml, device=device)
                model_results["accuracy"] = accuracy
                print(f"  mAP@50: {accuracy['mAP50']:.4f}")
                print(f"  mAP@50-95: {accuracy['mAP50-95']:.4f}")
                print(f"  Precision: {accuracy['precision']:.4f}")
                print(f"  Recall: {accuracy['recall']:.4f}")
            else:
                print(f"  跳过: 数据集不存在")
                model_results["accuracy"] = {"error": "数据集不存在"}
        except Exception as e:
            print(f"  错误: {e}")
            model_results["accuracy"] = {"error": str(e)}

        results["models"][model_path] = model_results

    # 保存结果
    result_file = output_path / "comparison_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {result_file}")

    # 打印汇总表格
    print_summary(results)

    return results


def print_summary(results: dict):
    """打印汇总表格"""
    print("\n" + "=" * 80)
    print("对比结果汇总")
    print("=" * 80)

    # 表头
    print(f"{'模型':<15} {'参数量':<12} {'大小(MB)':<10} {'FPS':<8} {'mAP50':<10} {'mAP50-95':<10}")
    print("-" * 80)

    for model_name, data in results["models"].items():
        params = data.get("info", {}).get("total_params", 0)
        size = data.get("info", {}).get("model_size_mb", 0)
        fps = data.get("speed", {}).get("fps", 0)
        map50 = data.get("accuracy", {}).get("mAP50", 0)
        map50_95 = data.get("accuracy", {}).get("mAP50-95", 0)

        params_str = f"{params/1e6:.1f}M" if params else "N/A"
        size_str = f"{size:.1f}" if size else "N/A"
        fps_str = f"{fps:.1f}" if fps else "N/A"
        map50_str = f"{map50:.4f}" if map50 else "N/A"
        map50_95_str = f"{map50_95:.4f}" if map50_95 else "N/A"

        print(f"{model_name:<15} {params_str:<12} {size_str:<10} {fps_str:<8} {map50_str:<10} {map50_95_str:<10}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="YOLO模型对比实验")
    parser.add_argument("--models", "-m", nargs="+", default=None,
                        help="要对比的模型列表")
    parser.add_argument("--data", "-d", default="datasets/traffic.yaml",
                        help="数据集配置文件")
    parser.add_argument("--device", default="cpu",
                        help="测试设备 (0, 1, cpu)")
    parser.add_argument("--output", "-o", default="experiments/comparison",
                        help="结果保存目录")
    args = parser.parse_args()

    run_comparison(
        models=args.models,
        data_yaml=args.data,
        device=args.device,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
