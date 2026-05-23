# 实时交通分析系统

基于 YOLOv12 + ByteTrack 的智能交通监控系统，实现车辆检测、跟踪、碰撞风险预测和自适应违规识别。

## 功能特性

### 核心功能

| 功能 | 描述 |
|------|------|
| **车辆检测** | YOLOv12 实时检测多种车型（小汽车、卡车、公交车、摩托车等） |
| **多目标跟踪** | ByteTrack 跨帧关联，保持轨迹连续 |
| **碰撞风险预测** | LSTM 轨迹预测 + TTC 分析 |
| **ST-GAT 车辆交互** | 时空图注意力网络建模车辆间关系 |
| **自适应违规检测** | 闯红灯、超速、逆行、违规变道检测 |
| **特种车辆识别** | 识别救护车、消防车、警车、工程救险车 |
| **车牌 OCR** | CRNN + PaddleOCR 双引擎车牌识别 |
| **数据库存储** | SQLite 结构化存储车辆和违规记录 |

### 创新点

1. **碰撞风险预测系统**
   - LSTM 轨迹预测，基于历史数据预测未来 15 帧位置
   - TTC（Time-To-Collision）计算
   - 跟车距离检测
   - 风险等级可视化（安全/低风险/中风险/高风险/危急）

2. **自适应违规检测**
   - 智能识别特殊场景
   - 违规时若附近有特种车辆或交警，标记为异常待人工复核
   - 异常情况单独保存截图，便于后续审核

3. **性能自适应降级**
   - FPS 监控 + 动态分辨率调整
   - 负载高时自动降低推理分辨率、跳帧处理
   - 分级降级策略，保障系统稳定性

4. **远程车辆避让检测**
   - ST-GAT 图注意力网络建模车辆交互
   - 识别车辆对特种车辆的避让行为

## 项目结构

```
yolo_ls/
├── src/
│   ├── core/                     # 核心模块
│   │   ├── detector.py           # YOLO 车辆检测 + 切片推理
│   │   ├── tracker.py            # ByteTrack 多目标跟踪
│   │   ├── collision_risk.py     # 碰撞风险预测（LSTM + TTC）
│   │   ├── stgat.py              # ST-GAT 时空图注意力网络
│   │   ├── adaptive_violation.py # 自适应违规检测
│   │   ├── emergency_vehicle.py  # 特种车辆检测
│   │   ├── feature.py            # 特征提取（颜色、速度、方向）
│   │   ├── lane_violation.py     # 车道违规分析
│   │   └── violation.py          # 基础违规检测
│   ├── video/                    # 视频处理模块
│   │   ├── stream.py             # 视频流接入
│   │   └── preprocessor.py       # 帧预处理
│   ├── ocr/                      # 车牌识别模块
│   │   ├── plate_reader.py       # 车牌检测 + OCR
│   │   └── ocr_scheduler.py      # OCR 调度器
│   ├── database/                 # 数据存储
│   │   ├── db.py                 # SQLite 数据库操作
│   │   └── scheduler.py          # 数据库清理调度
│   ├── gui/                      # PyQt5 界面
│   │   ├── main_window.py        # 主窗口
│   │   ├── widgets/              # UI 组件
│   │   ├── theme.py              # 主题配置
│   │   └── i18n.py               # 国际化
│   └── utils/                    # 工具函数
│       ├── config.py             # 配置管理
│       ├── config_schema.py      # 配置验证
│       ├── performance.py        # 性能优化
│       ├── logging_config.py     # 日志配置
│       └── model_paths.py        # 模型路径管理
├── scripts/                      # 工具脚本
│   ├── batch_detect.py           # 批量检测（含碰撞风险）
│   ├── train.py                  # YOLO 检测模型训练
│   ├── train_collision_predictor.py # 碰撞轨迹预测模型训练
│   ├── train_stgat.py            # ST-GAT 交互模型训练
│   ├── evaluate_project.py       # 项目评估脚本
│   └── download_models.py        # 模型下载
├── tests/                        # 测试文件
│   ├── test_adaptive_violation.py
│   ├── test_lane_violation.py
│   ├── test_stgat_collision.py
│   ├── test_config_schema.py
│   └── test_video_stream.py
├── config/
│   └── settings.yaml             # 系统配置
├── models/                       # 模型权重文件
├── data/                         # 数据目录
│   ├── snapshots/                # 违规截图
│   │   ├── violations/           # 正常违规
│   │   └── anomaly/              # 异常情况（待人工复核）
│   └── traffic.db                # SQLite 数据库
├── docs/
│   ├── evaluation/               # 评估报告
│   └── mid-term-report/          # 中期报告
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.sh
└── main.py                       # 程序入口
```

## 技术栈

| 类别 | 技术 |
|------|------|
| **深度学习** | PyTorch, YOLOv12 (Ultralytics) |
| **目标跟踪** | ByteTrack |
| **视频处理** | OpenCV |
| **轨迹预测** | LSTM |
| **图神经网络** | ST-GAT |
| **OCR** | CRNN + PaddleOCR |
| **GUI** | PyQt5 |
| **数据库** | SQLite |
| **容器化** | Docker |
| **测试** | pytest |

## 快速开始

### 方式一：本地 Python 环境（推荐）

```bash
# 1. 克隆项目
cd yolo_ls

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 准备模型权重
python scripts/download_models.py

# 6. 运行 GUI
python main.py --gui

# 或处理视频文件
python main.py --source video.mp4

# 或使用摄像头
python main.py --source 0

# 无头模式（仅处理，不显示窗口）
python main.py --headless
```

### 方式二：Docker（推荐）

#### 快速开始

```bash
# 1. 复制环境变量配置文件
cp .env.example .env

# 2. 下载模型（可选）
docker compose --profile download up

# 3. 启动 GUI 模式
docker compose up

# 4. 或启动命令行模式
docker compose --profile cli up

# 5. 或进行批量检测
docker compose --profile batch up
```

#### 常用命令

| 命令 | 功能 |
|------|------|
| `docker compose up` | 启动 GUI 服务 |
| `docker compose --profile cli up` | 启动命令行模式 |
| `docker compose --profile batch up` | 启动批量检测 |
| `docker compose --profile download up` | 下载模型权重 |
| `docker compose build` | 重新构建镜像 |
| `docker compose down` | 停止并删除容器 |

#### 环境变量配置

可在 `.env` 文件中自定义以下变量：

- `VIDEO_SOURCE`：视频源路径（默认为 `/app/videos/test.mp4`）
- `VIDEO_OUTPUT`：输出视频路径（默认为 `/app/data/output.mp4`）
- `DEVICE`：运行设备（`cuda` 或 `cpu`，默认为 `cuda`）
- `INPUT_DIR`：批量检测输入目录（默认为 `/app/videos`）
- `OUTPUT_DIR`：批量检测输出目录（默认为 `/app/data/output`）
- `SKIP_FRAMES`：批量检测抽帧间隔（默认为 `10`）
- `DISPLAY`：X11 显示地址（仅 Linux，默认为 `:0`）

#### 原生命令（替代方案）

```bash
# 构建镜像
docker build -t traffic-analysis:latest .

# 运行 GUI（Linux）
xhost +local:docker
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/app \
  traffic-analysis:latest

# 运行 GUI（Windows，需安装 VcXsrv）
docker run --rm -e DISPLAY=host.docker.internal:0 -v ${PWD}:/app traffic-analysis:latest

# 批量检测（无需 GUI）
docker run --rm -v ${PWD}:/app traffic-analysis:latest \
  python scripts/batch_detect.py -i /app/video.mp4 -o /app/output -n 10
```

### CLI 参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--gui` | 启动图形界面模式 | False |
| `--source` | 视频源（摄像头 ID、RTSP 地址或视频文件路径） | "0" |
| `--model` | YOLO 模型路径 | "models/yolo12n_vehicle.pt" |
| `--confidence` | 检测置信度阈值 | 0.2 |
| `--device` | 运行设备（cuda/cpu） | "cuda" |
| `--config` | 配置文件路径 | "config/settings.yaml" |
| `--output` | 输出视频路径 | None |
| `--risk-output` | 输出逐帧结构化结果（JSONL）路径 | None |
| `--collision-model` | 碰撞预测模型权重路径 | None |
| `--stgat-model` | ST-GAT 模型权重路径 | None |
| `--enable-risk` | 强制启用碰撞风险检测 | False |
| `--disable-risk` | 强制关闭碰撞风险检测 | False |
| `--headless` | 无头模式（不显示窗口） | False |

## 配置说明

系统配置文件位于 [config/settings.yaml](file:///d:/Personal/DP/yolov1/yolo_ls-main/config/settings.yaml)，主要配置项：

```yaml
# 系统配置
system:
  device: "cuda"  # cuda 或 cpu
  log_level: "INFO"

# 性能优化
performance:
  enabled: true
  target_fps: 15
  dynamic_resolution: true

# YOLO 检测配置
detector:
  model_path: "models/yolo12n_vehicle.pt"
  confidence: 0.2
  imgsz: 768
  enable_tiling: false  # 切片推理，提升远处小目标检测

# 违规检测配置
violation:
  speed_limit: 60
  expected_flow_direction: "south"
  stop_line:
    y: 430
    x_start: 200
    x_end: 1100

# 风险预测配置
risk:
  enabled: true
  collision_threshold: 150.0
  ttc_thresholds:
    critical: 0.5
    high: 1.0
    medium: 2.0
    low: 3.0
```

## 性能指标

| 指标 | 目标值 |
|------|--------|
| 实时处理帧率 | ≥10 fps |
| 车辆检测准确率 | ≥85% |
| 碰撞预测提前时间 | 1-3 秒 |
| 连续运行稳定性 | 72 小时无崩溃 |

## 模型落地

### 训练检测模型（外部数据集微调）

```bash
python scripts/train.py \
  --model yolo12m.pt \
  --data datasets/vehicle_detection/data.yaml \
  --epochs 100 \
  --batch 16 \
  --device 0
```

### 训练碰撞轨迹预测模型（LSTM）

```bash
python scripts/train_collision_predictor.py \
  --source /path/to/video_or_dir \
  --model models/yolo12n_vehicle.pt \
  --history-length 10 \
  --prediction-horizon 15 \
  --epochs 20 \
  --device cpu
```

### 训练 ST-GAT 交互模型

```bash
python scripts/train_stgat.py \
  --source /path/to/video_or_dir \
  --model models/yolo12n_vehicle.pt \
  --epochs 20 \
  --device cpu
```

### 实时推理并导出结构化结果

```bash
python main.py \
  --source 0 \
  --model models/yolo12n_vehicle.pt \
  --collision-model experiments/collision_predictor_xxx/best.pt \
  --stgat-model experiments/stgat_xxx/best.pt \
  --risk-output data/realtime_events.jsonl \
  --output data/realtime_demo.mp4
```

## 核心模块 API

### 1. 车辆检测 ([detector.py](file:///d:/Personal/DP/yolov1/yolo_ls-main/src/core/detector.py))

```python
from src.core import VehicleDetector

detector = VehicleDetector(model_path='models/yolo12n_vehicle.pt', confidence=0.2)
detections = detector.detect_vehicles(frame)
# 返回: [Detection(bbox, confidence, class_name), ...]
```

### 2. 多目标跟踪 ([tracker.py](file:///d:/Personal/DP/yolov1/yolo_ls-main/src/core/tracker.py))

```python
from src.core import ByteTracker

tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
tracks = tracker.update(detections)
# 返回: [Track(track_id, bbox, class_name), ...]
```

### 3. 碰撞风险预测 ([collision_risk.py](file:///d:/Personal/DP/yolov1/yolo_ls-main/src/core/collision_risk.py))

```python
from src.core.collision_risk import CollisionRiskPredictor

predictor = CollisionRiskPredictor(fps=15.0)
risks = predictor.update(tracks)
# 返回: [CollisionRisk(vehicle1_id, vehicle2_id, risk_level, ttc), ...]
```

### 4. 自适应违规检测 ([adaptive_violation.py](file:///d:/Personal/DP/yolov1/yolo_ls-main/src/core/adaptive_violation.py))

```python
from src.core.adaptive_violation import AdaptiveViolationDetector

detector = AdaptiveViolationDetector(speed_limit=60)
detector.set_stop_line(y=430, x_start=200, x_end=1100)
record = detector.check_violation(track_id, bbox, speed, frame)
```

## 开发指南

### 运行测试

```bash
pytest tests/
```

### 评估项目

```bash
python scripts/evaluate_project.py --ocr-samples 150 --bench-samples 100 --device cpu
```

### 代码规范

- 遵循 PEP 8 规范
- 使用类型注解
- 函数/类必须有 docstring
- 变量命名：snake_case
- 类命名：PascalCase
- 常量命名：UPPER_SNAKE_CASE

## 许可证

本项目仅供学习和研究使用。

## 作者

GitHub: https://github.com/he523/yolo
