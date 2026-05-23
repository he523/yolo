# Real-Time Traffic Analysis System - Development Guidelines

## Project Overview
基于 YOLOv12 + ByteTrack 的实时交通分析系统，实现车辆检测、跟踪、碰撞风险预测、自适应违规识别和特种车辆避让检测。

## Repository
- **GitHub**: https://github.com/he523/yolo
- **Author**: he523

## Defaults

- Reply in **Chinese** unless I explicitly ask for English.
- No emojis.
- 没有我的允许不能修改此claude.md文件
- Do not truncate important outputs (logs, diffs, stack traces, commands,or critical reasoning that affects safety/correctness).
  
## Tech Stack
- **Deep Learning**: PyTorch, YOLOv12 (Ultralytics)
- **Video Processing**: OpenCV
- **Object Tracking**: ByteTrack
- **OCR**: CRNN / PaddleOCR
- **GUI**: PyQt5
- **Database**: SQLite
- **Visualization**: Matplotlib
- **Graph Neural Network**: ST-GAT
- **Trajectory Prediction**: LSTM
- **Containerization**: Docker
- **Testing**: pytest

## Project Structure
```
yolo_ls/
├── src/
│   ├── core/                 # 核心模块
│   │   ├── detector.py       # YOLO 目标检测 + 切片推理
│   │   ├── tracker.py        # ByteTrack 多目标跟踪
│   │   ├── collision_risk.py # LSTM 碰撞风险预测
│   │   ├── stgat.py          # ST-GAT 时空图注意力网络
│   │   ├── adaptive_violation.py # 自适应违规检测
│   │   ├── emergency_vehicle.py  # 特种车辆检测
│   │   ├── feature.py        # 特征提取（颜色、速度、方向）
│   │   ├── lane_violation.py # 车道违规分析
│   │   ├── robust_color.py   # 鲁棒颜色检测
│   │   └── violation.py      # 基础违规检测
│   ├── ocr/                  # 车牌识别模块
│   │   ├── plate_reader.py   # 车牌检测 + OCR
│   │   └── ocr_scheduler.py  # OCR 调度器
│   ├── video/                # 视频处理模块
│   │   ├── stream.py         # 视频流接入
│   │   └── preprocessor.py   # 帧预处理
│   ├── database/             # 数据存储
│   │   ├── db.py             # SQLite 数据库操作
│   │   └── scheduler.py      # 数据库清理调度
│   ├── gui/                  # PyQt5 界面
│   │   ├── main_window.py    # 主窗口
│   │   ├── widgets/          # UI 组件
│   │   ├── theme.py          # 主题配置
│   │   └── i18n.py           # 国际化
│   └── utils/                # 工具函数
│       ├── config.py         # 配置管理
│       ├── config_schema.py  # 配置验证
│       ├── constants.py      # 常量定义
│       ├── performance.py    # 性能优化（自适应降级）
│       ├── logging_config.py # 日志配置
│       └── model_paths.py    # 模型路径管理
├── scripts/                  # 工具脚本
│   ├── train.py              # YOLO 检测模型训练
│   ├── train_collision_predictor.py # LSTM 轨迹预测训练
│   ├── train_stgat.py        # ST-GAT 交互模型训练
│   ├── train_plate_ocr.py    # 车牌 OCR 训练
│   ├── batch_detect.py       # 批量检测（含碰撞风险）
│   ├── evaluate_project.py   # 项目评估脚本
│   ├── download_models.py    # 模型下载
│   └── visualize.py          # 可视化工具
├── tests/                    # 测试文件
│   ├── test_adaptive_violation.py
│   ├── test_lane_violation.py
│   ├── test_stgat_collision.py
│   ├── test_config_schema.py
│   └── test_video_stream.py
├── config/                   # 配置文件
│   └── settings.yaml         # 系统配置
├── data/                     # 数据目录
│   ├── snapshots/            # 违规截图
│   └── traffic.db            # SQLite 数据库
├── models/                   # 模型权重文件
├── docs/                     # 文档
│   ├── evaluation/           # 评估报告
│   └── mid-term-report/      # 中期报告
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── main.py                   # 程序入口
```

## Coding Standards

### Python Style
- 遵循 PEP 8 规范
- 使用 type hints
- 函数/类必须有 docstring
- 变量命名：snake_case
- 类命名：PascalCase
- 常量命名：UPPER_SNAKE_CASE

### Code Example
```python
def detect_vehicles(frame: np.ndarray, confidence: float = 0.5) -> List[Detection]:
    """
    检测图像中的车辆。

    Args:
        frame: BGR 格式的图像帧
        confidence: 置信度阈值

    Returns:
        检测结果列表
    """
    pass
```

## Performance Requirements
- 实时处理帧率: ≥10 fps
- 车辆检测准确率: ≥85%
- 颜色识别准确率: ≥90%
- 速度计算误差: ≤5 km/h
- 碰撞预测提前时间: 1-3秒
- 连续运行: 72小时无崩溃

## Module Specifications

### 1. Video Input Module
- 支持 RTSP 流（摄像头）
- 支持本地视频文件（mp4, avi）
- 支持 DVR/NVR 接入
- 帧率控制：10-15 fps
- 分辨率标准化：640×640

### 2. Detection Module (YOLOv12)
- 检测目标：车辆、车牌、交通灯、停止线
- 预训练模型：COCO
- 微调数据集：UA-DETRAC, KITTI
- 输出格式：[x1, y1, x2, y2, confidence, class_id]
- 支持切片推理提升远处小目标检测

### 3. Tracking Module (ByteTrack)
- 为每个车辆分配唯一 ID
- 跨帧关联
- 轨迹记录（50帧历史）
- 卡尔曼滤波预测

### 4. Feature Analysis Module
- **颜色识别**: RGB → HSV, 光照鲁棒检测
- **速度计算**: 像素位移 → 实际距离（需相机标定）
- **方向判断**: 位移向量分析（8方向）
- **车型分类**: 轿车/SUV/卡车

### 5. Collision Risk Module
- LSTM 轨迹预测（预测未来15帧）
- TTC (Time-To-Collision) 计算
- 跟车距离检测
- 多级风险评估（安全/低/中/高/危急）

### 6. Violation Detection Module
- **闯红灯**: 交通灯状态 + 停止线位置 + 车辆轨迹
- **超速**: 速度阈值判断
- **逆行**: 方向与预期流向对比
- **违规变道**: 横向位移检测

### 7. Adaptive Violation Detection
- 特种车辆识别（救护车、消防车、警车）
- 交警检测
- 异常情况标记（待人工复核）
- 信号灯故障检测

### 8. ST-GAT Module
- 时空图注意力网络
- 车辆交互建模
- 避让行为识别

### 9. OCR Module
- 车牌定位（基于颜色和形态学）
- CRNN 字符识别
- PaddleOCR 兜底
- 格式校验（中国车牌格式）

### 10. GUI Module (PyQt5)
- 实时监控画面
- 统计图表（流量趋势、违规分布）
- 数据检索（按车牌/时间）
- 违规告警
- 主题切换（深色/浅色）

### 11. Database Module (SQLite)
Tables:
- `vehicles`: 车辆记录（track_id, plate_number, color, speed, direction）
- `violations`: 违规记录（类型、位置、截图、免责标记）
- `traffic_flow`: 流量统计（车辆数、平均速度、方向）

### 12. Performance Optimization
- FPS 监控与自适应降级
- 动态分辨率调整
- 跳帧处理
- 推理间隔控制

## Development Phases

### Phase 1: Foundation
- [x] 项目框架搭建
- [x] 视频流接入
- [x] YOLO 检测集成

### Phase 2: Core Features
- [x] ByteTrack 跟踪
- [x] 颜色/速度/方向分析
- [x] 车牌 OCR

### Phase 3: Violation Detection
- [x] 闯红灯检测
- [x] 超速检测
- [x] 逆行检测
- [x] 违规变道检测

### Phase 4: Advanced Features
- [x] 碰撞风险预测（LSTM）
- [x] ST-GAT 车辆交互建模
- [x] 特种车辆检测
- [x] 自适应违规检测

### Phase 5: GUI & Database
- [x] PyQt5 界面
- [x] SQLite 数据存储
- [x] 数据可视化

### Phase 6: Testing & Optimization
- [x] 性能优化（自适应降级）
- [x] 测试验证
- [x] 文档完善

## Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
PyQt5>=5.15.0
matplotlib>=3.7.0
paddleocr>=2.7.0
paddlepaddle>=2.5.0
pyyaml>=6.0
lap>=0.4.0
scipy>=1.10.0
filterpy>=1.4.5
apscheduler>=3.10.0
pytest>=7.0.0
```

## Notes
- GPU 加速优先（CUDA），支持自动回退到 CPU
- 异常处理要完善，关键操作需日志记录
- 配置文件统一管理，支持动态修改
- 模块间低耦合，便于独立测试和替换
- 支持 Docker 容器化部署
- 性能监控与自适应降级保障系统稳定性

## Deployment
```bash
# 本地运行
python main.py --gui

# 命令行模式
python main.py --source video.mp4 --headless

# Docker 部署
docker compose up

# 批量检测
docker compose --profile batch up
```
