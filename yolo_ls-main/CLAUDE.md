# Real-Time Traffic Analysis System - Development Guidelines

## Project Overview
基于 YOLOv12 + ByteTrack 的实时交通分析系统，实现车辆检测、跟踪、特征提取、违规识别等功能。

## Repository
- **GitHub**: https://github.com/Zhye26/yolo_ls
- **Author**: Zhye26

## Defaults

- Reply in **Chinese** unless I explicitly ask for English.
- No emojis.
- 没有我的允许不能修改此claude.md文件
- Do not truncate important outputs (logs, diffs, stack traces, commands,or critical reasoning that affects safety/correctness).
  
## Tech Stack
- **Deep Learning**: PyTorch, YOLOv12 (Ultralytics)
- **Video Processing**: OpenCV
- **Object Tracking**: ByteTrack
- **OCR**: Tesseract / PaddleOCR
- **GUI**: PyQt5
- **Database**: SQLite
- **Visualization**: Matplotlib, Seaborn

## Project Structure
```
yolo_ls/
├── src/
│   ├── core/                 # 核心模块
│   │   ├── detector.py       # YOLO 目标检测
│   │   ├── tracker.py        # ByteTrack 跟踪
│   │   ├── feature.py        # 特征提取（颜色、速度、方向）
│   │   └── violation.py      # 违规检测（闯红灯、超速）
│   ├── ocr/                  # 车牌识别模块
│   │   └── plate_reader.py
│   ├── video/                # 视频处理模块
│   │   ├── stream.py         # 视频流接入
│   │   └── preprocessor.py   # 帧预处理
│   ├── database/             # 数据存储
│   │   └── db.py
│   ├── gui/                  # PyQt5 界面
│   │   ├── main_window.py
│   │   └── widgets/
│   └── utils/                # 工具函数
│       └── config.py
├── models/                   # 模型权重文件
├── data/                     # 数据集
├── tests/                    # 测试文件
├── config/                   # 配置文件
│   └── settings.yaml
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

### 3. Tracking Module (ByteTrack)
- 为每个车辆分配唯一 ID
- 跨帧关联
- 轨迹记录

### 4. Feature Analysis Module
- **颜色识别**: RGB → HSV, K-Means 聚类
- **速度计算**: 像素位移 → 实际距离（需相机标定）
- **方向判断**: 位移向量分析
- **车型分类**: 轿车/SUV/卡车

### 5. Violation Detection Module
- **闯红灯**: 交通灯状态 + 停止线位置 + 车辆轨迹
- **超速**: 速度阈值判断

### 6. OCR Module
- 车牌定位
- 字符分割
- 字符识别
- 格式校验（中国车牌格式）

### 7. GUI Module (PyQt5)
- 实时监控画面
- 统计图表（流量趋势、违规分布）
- 数据检索
- 违规告警

### 8. Database Module (SQLite)
Tables:
- `vehicles`: 车辆记录
- `violations`: 违规记录
- `traffic_flow`: 流量统计

## Development Phases

### Phase 1: Foundation
- [ ] 项目框架搭建
- [ ] 视频流接入
- [ ] YOLO 检测集成

### Phase 2: Core Features
- [ ] ByteTrack 跟踪
- [ ] 颜色/速度/方向分析
- [ ] 车牌 OCR

### Phase 3: Violation Detection
- [ ] 闯红灯检测
- [ ] 超速检测

### Phase 4: GUI & Database
- [ ] PyQt5 界面
- [ ] SQLite 数据存储
- [ ] 数据可视化

### Phase 5: Testing & Optimization
- [ ] 性能优化
- [ ] 测试验证
- [ ] 文档完善

## Dependencies
```
torch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
PyQt5>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytesseract>=0.3.10
```

## Notes
- GPU 加速优先（CUDA）
- 异常处理要完善
- 日志记录关键操作
- 配置文件统一管理
- 模块间低耦合
