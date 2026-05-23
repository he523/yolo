"""
完整测试数据集生成脚本
生成用于交通分析系统的各类测试数据：
1. 车辆检测数据集 (YOLO格式)
2. 碰撞预测数据集
3. ST-GAT车辆交互数据集
4. 车牌识别数据集
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
import random
from tqdm import tqdm
import yaml

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.detector import VehicleDetector


@dataclass
class VehicleType:
    name: str
    class_id: int
    color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    size_range: Tuple[Tuple[int, int], Tuple[int, int]]  # (min_w, min_h), (max_w, max_h)
    speed_range: Tuple[float, float]  # km/h


# 车辆类型定义
VEHICLE_TYPES = [
    VehicleType("car", 0, ((0, 0, 0), (255, 255, 255)), ((60, 40), (150, 100)), (30, 120)),
    VehicleType("motorcycle", 1, ((0, 0, 0), (255, 255, 255)), ((30, 20), (60, 40)), (20, 80)),
    VehicleType("bus", 2, ((200, 200, 200), (255, 255, 255)), ((120, 80), (250, 150)), (20, 80)),
    VehicleType("truck", 3, ((100, 100, 100), (200, 200, 200)), ((100, 70), (200, 140)), (20, 90)),
]

# 车牌格式
PLATE_FORMATS = [
    "京A{}{}{}{}{}{}",
    "沪B{}{}{}{}{}{}",
    "粤C{}{}{}{}{}{}",
    "苏D{}{}{}{}{}{}",
    "浙E{}{}{}{}{}{}",
]

# 车牌字符集
CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"


class SyntheticDataGenerator:
    """合成数据生成器"""

    def __init__(self, output_dir: str = "datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 道路背景参数
        self.road_colors = [
            (50, 50, 50), (60, 60, 60), (70, 70, 70),  # 沥青路
            (180, 180, 180), (190, 190, 190),  # 水泥路
        ]
        self.lane_colors = [(255, 255, 255), (200, 200, 200)]
        self.grass_colors = [(34, 139, 34), (46, 139, 87), (50, 170, 50)]

    def generate_road_background(self, width: int = 1280, height: int = 720) -> np.ndarray:
        """生成道路背景"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 天空
        sky_color = (135, 206, 235)  # 天蓝色
        img[:height//3, :] = sky_color
        
        # 草地
        grass_color = random.choice(self.grass_colors)
        img[height//3:height//2, :] = grass_color
        
        # 道路
        road_color = random.choice(self.road_colors)
        img[height//2:, :] = road_color
        
        # 车道线
        lane_y1 = height//2 + height//6
        lane_y2 = height//2 + height//3
        
        # 中心黄线
        cv2.line(img, (width//2, lane_y1), (width//2, height), (0, 200, 255), 3)
        
        # 白色虚线
        for y in range(lane_y1, height, 50):
            for x in range(100, width, 200):
                cv2.line(img, (x, y), (x + 80, y), (255, 255, 255), 3)
        
        # 路缘
        cv2.line(img, (0, height//2), (width, height//2), (100, 100, 100), 2)
        
        return img

    def generate_vehicle(self, img: np.ndarray, x: int, y: int, vehicle_type: VehicleType) -> Dict:
        """在图像上生成车辆"""
        h, w = img.shape[:2]
        
        # 随机尺寸
        (min_w, min_h), (max_w, max_h) = vehicle_type.size_range
        veh_w = random.randint(min_w, max_w)
        veh_h = random.randint(min_h, max_h)
        
        # 确保不超出边界
        x = max(0, min(x, w - veh_w))
        y = max(0, min(y, h - veh_h))
        
        # 随机颜色
        color = (
            random.randint(vehicle_type.color_range[0][0], vehicle_type.color_range[1][0]),
            random.randint(vehicle_type.color_range[0][1], vehicle_type.color_range[1][1]),
            random.randint(vehicle_type.color_range[0][2], vehicle_type.color_range[1][2]),
        )
        
        # 绘制车辆主体（矩形）
        cv2.rectangle(img, (x, y), (x + veh_w, y + veh_h), color, -1)
        
        # 绘制车窗
        window_y = y + int(veh_h * 0.2)
        window_h = int(veh_h * 0.4)
        cv2.rectangle(img, (x + int(veh_w * 0.1), window_y), 
                     (x + int(veh_w * 0.9), window_y + window_h), 
                     (100, 150, 200), -1)
        
        # 绘制车轮
        wheel_r = int(veh_h * 0.15)
        wheel_y = y + veh_h
        cv2.circle(img, (x + int(veh_w * 0.2), wheel_y - wheel_r), wheel_r, (30, 30, 30), -1)
        cv2.circle(img, (x + int(veh_w * 0.8), wheel_y - wheel_r), wheel_r, (30, 30, 30), -1)
        
        # 生成车牌
        plate = self.generate_license_plate()
        
        # 返回车辆信息
        return {
            "class_id": vehicle_type.class_id,
            "class_name": vehicle_type.name,
            "bbox": [x, y, x + veh_w, y + veh_h],
            "color": color,
            "speed": random.uniform(*vehicle_type.speed_range),
            "plate": plate,
        }

    def generate_license_plate(self) -> str:
        """生成车牌号"""
        format_str = random.choice(PLATE_FORMATS)
        chars = [random.choice(CHARS) for _ in range(6)]
        return format_str.format(*chars)

    def generate_detection_dataset(self, num_train: int = 500, num_val: int = 100):
        """生成车辆检测数据集 (YOLO格式)"""
        print("\n" + "=" * 60)
        print("生成车辆检测数据集...")
        print("=" * 60)
        
        dataset_dir = self.output_dir / "vehicle_detection"
        
        # 创建目录结构
        for split in ["train", "val"]:
            (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # 生成数据
        splits = [("train", num_train), ("val", num_val)]
        
        for split, count in splits:
            print(f"\n生成 {split} 集 ({count} 张)...")
            
            for idx in tqdm(range(count), desc=f"{split}"):
                # 生成背景
                img = self.generate_road_background()
                h, w = img.shape[:2]
                
                # 生成车辆
                labels = []
                num_vehicles = random.randint(2, 8)
                
                for _ in range(num_vehicles):
                    vehicle_type = random.choice(VEHICLE_TYPES)
                    x = random.randint(50, w - 200)
                    y = random.randint(h//2 + 50, h - 150)
                    
                    vehicle = self.generate_vehicle(img, x, y, vehicle_type)
                    
                    # YOLO格式: class_id cx cy nw nh (归一化)
                    x1, y1, x2, y2 = vehicle["bbox"]
                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    nw = (x2 - x1) / w
                    nh = (y2 - y1) / h
                    
                    labels.append(f"{vehicle_type.class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
                # 保存图片
                img_name = f"det_{split}_{idx:05d}.jpg"
                cv2.imwrite(str(dataset_dir / "images" / split / img_name), img)
                
                # 保存标注
                label_name = img_name.replace(".jpg", ".txt")
                with open(dataset_dir / "labels" / split / label_name, "w") as f:
                    f.write("\n".join(labels))
        
        # 生成YOLO配置文件
        self._generate_yolo_yaml(dataset_dir)
        
        print(f"\n检测数据集生成完成!")
        print(f"  训练集: {num_train} 张")
        print(f"  验证集: {num_val} 张")

    def _generate_yolo_yaml(self, dataset_dir: Path):
        """生成YOLO数据集配置文件"""
        config = {
            "path": str(dataset_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "car",
                1: "motorcycle", 
                2: "bus",
                3: "truck",
            }
        }
        
        with open(dataset_dir / "data.yaml", "w") as f:
            yaml.dump(config, f, allow_unicode=True)

    def generate_collision_dataset(self, num_sequences: int = 50, frames_per_seq: int = 60):
        """生成碰撞预测数据集"""
        print("\n" + "=" * 60)
        print("生成碰撞预测数据集...")
        print("=" * 60)
        
        dataset_dir = self.output_dir / "collision_prediction"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        sequences = []
        
        print(f"\n生成 {num_sequences} 个序列，每个 {frames_per_seq} 帧...")
        
        for seq_idx in tqdm(range(num_sequences), desc="序列"):
            seq_data = {
                "sequence_id": f"seq_{seq_idx:04d}",
                "frames": [],
                "collision_risk": []
            }
            
            # 初始化车辆轨迹
            vehicles = []
            for veh_id in range(random.randint(3, 6)):
                vehicle_type = random.choice(VEHICLE_TYPES)
                start_x = random.randint(100, 300)
                start_y = random.randint(400, 600)
                
                # 随机运动方向和速度
                dx = random.uniform(-3, 5)  # 水平速度
                dy = random.uniform(-0.5, 0.5)  # 垂直速度
                
                vehicles.append({
                    "id": veh_id,
                    "type": vehicle_type,
                    "x": start_x,
                    "y": start_y,
                    "vx": dx,
                    "vy": dy,
                    "history": []
                })
            
            # 生成帧序列
            for frame_idx in range(frames_per_seq):
                img = self.generate_road_background()
                frame_data = {
                    "frame_idx": frame_idx,
                    "vehicles": []
                }
                
                # 更新车辆位置
                for veh in vehicles:
                    veh["x"] += veh["vx"]
                    veh["y"] += veh["vy"]
                    
                    # 边界反弹
                    h, w = img.shape[:2]
                    if veh["x"] < 50 or veh["x"] > w - 150:
                        veh["vx"] *= -1
                    if veh["y"] < h//2 + 50 or veh["y"] > h - 150:
                        veh["vy"] *= -1
                    
                    # 绘制车辆
                    vehicle_info = self.generate_vehicle(img, int(veh["x"]), int(veh["y"]), veh["type"])
                    
                    # 保存轨迹历史
                    veh["history"].append((veh["x"], veh["y"]))
                    if len(veh["history"]) > 20:
                        veh["history"].pop(0)
                    
                    frame_data["vehicles"].append({
                        "id": veh["id"],
                        "type": veh["type"].name,
                        "bbox": vehicle_info["bbox"],
                        "position": [veh["x"], veh["y"]],
                        "velocity": [veh["vx"], veh["vy"]],
                        "history": veh["history"].copy()
                    })
                
                seq_data["frames"].append(frame_data)
                
                # 保存帧
                seq_dir = dataset_dir / seq_data["sequence_id"]
                seq_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(seq_dir / f"frame_{frame_idx:04d}.jpg"), img)
            
            # 计算碰撞风险（简化版本）
            risks = []
            for frame in seq_data["frames"]:
                frame_risks = []
                vehs = frame["vehicles"]
                for i in range(len(vehs)):
                    for j in range(i + 1, len(vehs)):
                        v1, v2 = vehs[i], vehs[j]
                        dist = np.sqrt((v1["position"][0] - v2["position"][0])**2 + 
                                      (v1["position"][1] - v2["position"][1])**2)
                        
                        if dist < 200:
                            frame_risks.append({
                                "veh1_id": v1["id"],
                                "veh2_id": v2["id"],
                                "distance": dist,
                                "risk_level": "low" if dist > 150 else "medium" if dist > 100 else "high"
                            })
                risks.append(frame_risks)
            
            seq_data["collision_risk"] = risks
            sequences.append(seq_data)
            
            # 保存序列数据
            with open(seq_dir / "data.json", "w") as f:
                json.dump(seq_data, f, indent=2)
        
        # 保存整体数据索引
        with open(dataset_dir / "dataset_info.json", "w") as f:
            json.dump({
                "num_sequences": num_sequences,
                "frames_per_sequence": frames_per_seq,
                "sequences": [s["sequence_id"] for s in sequences]
            }, f, indent=2)
        
        print(f"\n碰撞预测数据集生成完成!")
        print(f"  序列数: {num_sequences}")
        print(f"  总帧数: {num_sequences * frames_per_seq}")

    def generate_stgat_dataset(self, num_sequences: int = 30, frames_per_seq: int = 100):
        """生成ST-GAT车辆交互数据集"""
        print("\n" + "=" * 60)
        print("生成ST-GAT车辆交互数据集...")
        print("=" * 60)
        
        dataset_dir = self.output_dir / "stgat_interaction"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        sequences = []
        
        print(f"\n生成 {num_sequences} 个交互场景...")
        
        for seq_idx in tqdm(range(num_sequences), desc="序列"):
            seq_data = {
                "sequence_id": f"stgat_{seq_idx:04d}",
                "frames": [],
                "interaction_type": random.choice(["normal", "yielding", "overtaking", "merging"])
            }
            
            # 生成场景
            for frame_idx in range(frames_per_seq):
                img = self.generate_road_background(width=960, height=540)
                h, w = img.shape[:2]
                
                # 生成车辆
                vehicles = []
                num_vehicles = random.randint(4, 10)
                
                for veh_id in range(num_vehicles):
                    vehicle_type = random.choice(VEHICLE_TYPES)
                    
                    # 根据交互类型生成位置
                    if seq_data["interaction_type"] == "yielding" and veh_id == 0:
                        # 特种车辆
                        x = w//2 - 100 + frame_idx * 2
                        y = h - 200
                    else:
                        x = random.randint(100, w - 200)
                        y = random.randint(h//2 + 50, h - 150)
                    
                    vehicle_info = self.generate_vehicle(img, x, y, vehicle_type)
                    
                    vehicles.append({
                        "id": veh_id,
                        "type": vehicle_type.name,
                        "class_id": vehicle_type.class_id,
                        "bbox": vehicle_info["bbox"],
                        "position": [(x + vehicle_info["bbox"][2])/2, (y + vehicle_info["bbox"][3])/2],
                        "is_emergency": veh_id == 0 and seq_data["interaction_type"] == "yielding"
                    })
                
                # 构建邻接矩阵
                adj_matrix = np.zeros((len(vehicles), len(vehicles)), dtype=np.float32)
                for i in range(len(vehicles)):
                    for j in range(len(vehicles)):
                        if i != j:
                            dist = np.sqrt(
                                (vehicles[i]["position"][0] - vehicles[j]["position"][0])**2 +
                                (vehicles[i]["position"][1] - vehicles[j]["position"][1])**2
                            )
                            if dist < 200:
                                adj_matrix[i][j] = 1.0 - (dist / 200)
                
                frame_data = {
                    "frame_idx": frame_idx,
                    "vehicles": vehicles,
                    "adjacency_matrix": adj_matrix.tolist()
                }
                
                seq_data["frames"].append(frame_data)
                
                # 保存帧
                seq_dir = dataset_dir / seq_data["sequence_id"]
                seq_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(seq_dir / f"frame_{frame_idx:04d}.jpg"), img)
            
            sequences.append(seq_data)
            
            # 保存序列数据
            with open(seq_dir / "data.json", "w") as f:
                json.dump(seq_data, f, indent=2)
        
        # 保存整体信息
        with open(dataset_dir / "dataset_info.json", "w") as f:
            json.dump({
                "num_sequences": num_sequences,
                "frames_per_sequence": frames_per_seq,
                "interaction_types": ["normal", "yielding", "overtaking", "merging"]
            }, f, indent=2)
        
        print(f"\nST-GAT数据集生成完成!")
        print(f"  序列数: {num_sequences}")
        print(f"  总帧数: {num_sequences * frames_per_seq}")

    def generate_plate_dataset(self, num_samples: int = 1000):
        """生成车牌识别数据集"""
        print("\n" + "=" * 60)
        print("生成车牌识别数据集...")
        print("=" * 60)
        
        dataset_dir = self.output_dir / "license_plate"
        (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
        
        samples = []
        
        print(f"\n生成 {num_samples} 个车牌样本...")
        
        for idx in tqdm(range(num_samples), desc="车牌"):
            # 生成车牌图片
            plate_img = self._generate_plate_image()
            plate_text = self.generate_license_plate()
            
            # 保存图片
            img_name = f"plate_{idx:05d}.jpg"
            cv2.imwrite(str(dataset_dir / "images" / img_name), plate_img)
            
            samples.append({
                "image": img_name,
                "text": plate_text,
            })
        
        # 保存标注
        with open(dataset_dir / "annotations.json", "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n车牌数据集生成完成!")
        print(f"  样本数: {num_samples}")

    def _generate_plate_image(self, width: int = 220, height: int = 70) -> np.ndarray:
        """生成车牌图片"""
        # 蓝底白字
        img = np.ones((height, width, 3), dtype=np.uint8) * (30, 100, 200)  # 蓝底
        
        # 添加边框
        cv2.rectangle(img, (2, 2), (width-3, height-3), (255, 255, 255), 2)
        
        # 生成车牌字符（简化版）
        plate_text = self.generate_license_plate()
        
        # 绘制文字（使用中文需要特殊字体，这里用英文模拟）
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
        
        # 居中绘制
        x = (width - text_size[0]) // 2
        y = (height + text_size[1]) // 2
        cv2.putText(img, plate_text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        # 添加噪声和模糊
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 随机亮度变化
        brightness = random.uniform(0.7, 1.3)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
        return img

    def generate_real_world_samples(self):
        """使用真实模型生成标注样本"""
        print("\n" + "=" * 60)
        print("使用预训练模型生成标注样本...")
        print("=" * 60)
        
        samples_dir = self.output_dir / "real_world_samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载一些公开的交通图片
        sample_urls = [
            "https://ultralytics.com/images/bus.jpg",
            "https://ultralytics.com/images/zidane.jpg",
        ]
        
        import urllib.request
        
        print("\n下载示例图片...")
        for idx, url in enumerate(sample_urls):
            try:
                img_path = samples_dir / f"real_{idx:02d}.jpg"
                urllib.request.urlretrieve(url, str(img_path))
                print(f"  下载: {img_path}")
            except Exception as e:
                print(f"  跳过: {e}")
        
        print(f"\n真实样本目录: {samples_dir}")

    def generate_all(self):
        """生成所有数据集"""
        print("\n" + "=" * 60)
        print("开始生成完整测试数据集")
        print("=" * 60)
        
        # 1. 车辆检测数据集
        self.generate_detection_dataset(num_train=200, num_val=50)
        
        # 2. 碰撞预测数据集
        self.generate_collision_dataset(num_sequences=20, frames_per_seq=50)
        
        # 3. ST-GAT交互数据集
        self.generate_stgat_dataset(num_sequences=15, frames_per_seq=80)
        
        # 4. 车牌识别数据集
        self.generate_plate_dataset(num_samples=500)
        
        # 5. 真实世界样本
        self.generate_real_world_samples()
        
        # 生成数据集说明
        self._generate_readme()
        
        print("\n" + "=" * 60)
        print("所有数据集生成完成!")
        print("=" * 60)
        print(f"\n数据集位置: {self.output_dir.absolute()}")

    def _generate_readme(self):
        """生成数据集说明文件"""
        readme_content = """# 交通分析系统测试数据集

## 数据集结构

```
datasets/
├── vehicle_detection/          # 车辆检测数据集 (YOLO格式)
│   ├── images/
│   │   ├── train/              # 训练集图片
│   │   └── val/                # 验证集图片
│   ├── labels/
│   │   ├── train/              # 训练集标注
│   │   └── val/                # 验证集标注
│   └── data.yaml               # YOLO配置文件
│
├── collision_prediction/       # 碰撞预测数据集
│   ├── seq_0000/
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── data.json           # 序列数据
│   └── dataset_info.json
│
├── stgat_interaction/          # ST-GAT车辆交互数据集
│   ├── stgat_0000/
│   │   ├── frame_0000.jpg
│   │   └── data.json
│   └── dataset_info.json
│
├── license_plate/              # 车牌识别数据集
│   ├── images/
│   └── annotations.json
│
└── real_world_samples/         # 真实世界样本
```

## 类别定义

- 0: car (小汽车)
- 1: motorcycle (摩托车)
- 2: bus (公交车)
- 3: truck (卡车)

## 使用方法

### 训练车辆检测模型
```bash
python scripts/train.py --model yolov12n.pt --data datasets/vehicle_detection/data.yaml --epochs 50
```

### 训练碰撞预测模型
```bash
python scripts/train_collision_predictor.py --source datasets/collision_prediction
```

### 训练ST-GAT模型
```bash
python scripts/train_stgat.py --source datasets/stgat_interaction
```

## 说明

本数据集包含合成数据和少量真实样本，用于系统测试和演示。
对于生产环境，建议使用真实标注的数据集，如UA-DETRAC、KITTI等。
"""
        
        with open(self.output_dir / "README.md", "w", encoding="utf-8") as f:
            f.write(readme_content)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="生成完整测试数据集")
    parser.add_argument("--output-dir", default="datasets", help="输出目录")
    parser.add_argument("--detection", action="store_true", help="只生成检测数据集")
    parser.add_argument("--collision", action="store_true", help="只生成碰撞预测数据集")
    parser.add_argument("--stgat", action="store_true", help="只生成ST-GAT数据集")
    parser.add_argument("--plate", action="store_true", help="只生成车牌数据集")
    parser.add_argument("--all", action="store_true", help="生成所有数据集 (默认)")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(output_dir=args.output_dir)
    
    if args.detection:
        generator.generate_detection_dataset()
    elif args.collision:
        generator.generate_collision_dataset()
    elif args.stgat:
        generator.generate_stgat_dataset()
    elif args.plate:
        generator.generate_plate_dataset()
    else:
        generator.generate_all()


if __name__ == "__main__":
    main()
