"""
UA-DETRAC 数据集下载和转换脚本
将UA-DETRAC格式转换为YOLO格式
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import urllib.request
import zipfile
import argparse


# UA-DETRAC类别映射到我们的类别
DETRAC_TO_YOLO = {
    'car': 0,
    'van': 0,      # van归类为car
    'bus': 2,
    'others': 3,   # others归类为truck
}


def download_sample_data(output_dir: str):
    """
    下载示例数据（用于快速测试）
    完整数据集需要从官网下载：https://detrac-db.rit.albany.edu/
    """
    print("=" * 50)
    print("UA-DETRAC 数据集说明")
    print("=" * 50)
    print()
    print("完整数据集需要手动下载：")
    print("1. 访问: https://detrac-db.rit.albany.edu/download")
    print("2. 下载以下文件：")
    print("   - DETRAC-train-data.zip (训练图片)")
    print("   - DETRAC-test-data.zip (测试图片)")
    print("   - DETRAC-Train-Annotations-XML.zip (训练标注)")
    print()
    print("3. 解压到 datasets/ 目录：")
    print("   datasets/")
    print("   ├── DETRAC-train-data/")
    print("   │   └── Insight-MVT_Annotation_Train/")
    print("   ├── DETRAC-test-data/")
    print("   │   └── Insight-MVT_Annotation_Test/")
    print("   └── DETRAC-Train-Annotations-XML/")
    print()
    print("4. 运行转换：")
    print("   python scripts/prepare_dataset.py --convert")
    print()

    # 创建示例数据用于测试流程
    print("正在创建示例数据用于测试...")
    create_sample_data(output_dir)


def create_sample_data(output_dir: str):
    """创建示例数据用于测试训练流程"""
    import cv2
    import numpy as np

    images_train = Path(output_dir) / "images" / "train"
    images_val = Path(output_dir) / "images" / "val"
    labels_train = Path(output_dir) / "labels" / "train"
    labels_val = Path(output_dir) / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # 下载一些真实的交通图片作为示例
    sample_urls = [
        ("https://ultralytics.com/images/bus.jpg", "bus"),
    ]

    print("下载示例图片...")
    for url, name in sample_urls:
        try:
            img_path = images_train / f"{name}.jpg"
            urllib.request.urlretrieve(url, str(img_path))

            # 复制到验证集
            shutil.copy(img_path, images_val / f"{name}.jpg")

            # 创建对应的标注文件（YOLO格式）
            # 格式: class_id center_x center_y width height (归一化)
            # bus.jpg 的标注（手动标注的近似值）
            if name == "bus":
                label_content = "2 0.5 0.5 0.9 0.7\n"  # bus
            else:
                label_content = "0 0.5 0.5 0.3 0.3\n"  # car

            with open(labels_train / f"{name}.txt", "w") as f:
                f.write(label_content)
            with open(labels_val / f"{name}.txt", "w") as f:
                f.write(label_content)

            print(f"  创建: {name}.jpg + {name}.txt")
        except Exception as e:
            print(f"  跳过 {name}: {e}")

    # 生成一些合成数据
    print("生成合成训练数据...")
    for i in range(20):
        # 创建随机背景
        img = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)

        # 添加一些矩形模拟车辆
        labels = []
        num_cars = np.random.randint(1, 5)
        for j in range(num_cars):
            x = np.random.randint(50, 550)
            y = np.random.randint(50, 550)
            w = np.random.randint(60, 150)
            h = np.random.randint(40, 100)

            # 随机颜色
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)

            # YOLO格式标注
            cx = (x + w/2) / 640
            cy = (y + h/2) / 640
            nw = w / 640
            nh = h / 640
            class_id = np.random.choice([0, 2, 3])  # car, bus, truck
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        # 保存
        split = "train" if i < 16 else "val"
        img_dir = images_train if split == "train" else images_val
        lbl_dir = labels_train if split == "train" else labels_val

        cv2.imwrite(str(img_dir / f"synthetic_{i:04d}.jpg"), img)
        with open(lbl_dir / f"synthetic_{i:04d}.txt", "w") as f:
            f.write("\n".join(labels))

    print(f"示例数据创建完成!")
    print(f"  训练集: {len(list(images_train.glob('*.jpg')))} 张")
    print(f"  验证集: {len(list(images_val.glob('*.jpg')))} 张")


def convert_detrac_to_yolo(detrac_dir: str, output_dir: str):
    """
    将UA-DETRAC数据集转换为YOLO格式

    Args:
        detrac_dir: UA-DETRAC数据集目录
        output_dir: 输出目录
    """
    detrac_path = Path(detrac_dir)
    output_path = Path(output_dir)

    # 检查目录
    train_images = detrac_path / "DETRAC-train-data" / "Insight-MVT_Annotation_Train"
    annotations = detrac_path / "DETRAC-Train-Annotations-XML"

    if not train_images.exists():
        print(f"错误: 找不到训练图片目录: {train_images}")
        print("请先下载UA-DETRAC数据集")
        return

    if not annotations.exists():
        print(f"错误: 找不到标注目录: {annotations}")
        return

    # 创建输出目录
    images_train = output_path / "images" / "train"
    images_val = output_path / "images" / "val"
    labels_train = output_path / "labels" / "train"
    labels_val = output_path / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取所有序列
    sequences = sorted([d for d in train_images.iterdir() if d.is_dir()])
    print(f"找到 {len(sequences)} 个视频序列")

    # 划分训练/验证集 (80/20)
    split_idx = int(len(sequences) * 0.8)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]

    total_images = 0
    total_labels = 0

    for split, seqs in [("train", train_seqs), ("val", val_seqs)]:
        img_dir = images_train if split == "train" else images_val
        lbl_dir = labels_train if split == "train" else labels_val

        for seq_dir in seqs:
            seq_name = seq_dir.name
            xml_file = annotations / f"{seq_name}.xml"

            if not xml_file.exists():
                print(f"  跳过 {seq_name}: 无标注文件")
                continue

            # 解析XML标注
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # 获取图片尺寸
            img_width = int(root.find(".//frame/target_list/target/box").get("width", 960))
            img_height = int(root.find(".//frame/target_list/target/box").get("height", 540))

            # 处理每一帧
            for frame in root.findall(".//frame"):
                frame_num = int(frame.get("num"))
                img_name = f"img{frame_num:05d}.jpg"
                src_img = seq_dir / img_name

                if not src_img.exists():
                    continue

                # 收集该帧的所有标注
                labels = []
                for target in frame.findall(".//target"):
                    box = target.find("box")
                    attr = target.find("attribute")

                    if box is None:
                        continue

                    # 获取类别
                    vehicle_type = attr.get("vehicle_type", "car") if attr is not None else "car"
                    if vehicle_type not in DETRAC_TO_YOLO:
                        vehicle_type = "car"
                    class_id = DETRAC_TO_YOLO[vehicle_type]

                    # 获取边界框
                    left = float(box.get("left"))
                    top = float(box.get("top"))
                    width = float(box.get("width"))
                    height = float(box.get("height"))

                    # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                    cx = (left + width / 2) / img_width
                    cy = (top + height / 2) / img_height
                    nw = width / img_width
                    nh = height / img_height

                    # 确保在有效范围内
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))

                    labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                if labels:
                    # 复制图片
                    dst_img = img_dir / f"{seq_name}_{img_name}"
                    shutil.copy(src_img, dst_img)
                    total_images += 1

                    # 保存标注
                    dst_lbl = lbl_dir / f"{seq_name}_{img_name}".replace(".jpg", ".txt")
                    with open(dst_lbl, "w") as f:
                        f.write("\n".join(labels))
                    total_labels += 1

            print(f"  处理完成: {seq_name}")

    print(f"\n转换完成!")
    print(f"  总图片数: {total_images}")
    print(f"  总标注数: {total_labels}")


def main():
    parser = argparse.ArgumentParser(description="准备交通检测数据集")
    parser.add_argument("--convert", action="store_true", help="转换UA-DETRAC数据集")
    parser.add_argument("--sample", action="store_true", help="创建示例数据")
    parser.add_argument("--detrac-dir", default="datasets", help="UA-DETRAC数据集目录")
    parser.add_argument("--output-dir", default="datasets", help="输出目录")
    args = parser.parse_args()

    if args.convert:
        convert_detrac_to_yolo(args.detrac_dir, args.output_dir)
    elif args.sample:
        create_sample_data(args.output_dir)
    else:
        download_sample_data(args.output_dir)


if __name__ == "__main__":
    main()
