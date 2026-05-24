#!/usr/bin/env python3
"""
红绿灯 ROI 数据采集工具

从视频/摄像头中逐帧播放，YOLO 自动检测红绿灯 bbox，
用户按键标注当前红绿灯状态（红/黄/绿/灭），
自动裁剪 ROI 并保存到对应分类文件夹。

按键说明:
  r - 红灯 (red)
  y - 黄灯 (yellow)
  g - 绿灯 (green)
  o - 灭灯/未知 (off)
  Space - 跳过当前帧
  d - 手动拖框标注（YOLO 未检出时使用）
  q - 退出
  s - 显示/隐藏统计
  +/- - 调整 ROI 外扩像素

数据集目录结构:
  datasets/traffic_light_cls/
    red/      # 红灯 ROI
    yellow/   # 黄灯 ROI
    green/    # 绿灯 ROI
    off/      # 灭灯/未知 ROI
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# 添加项目根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.detector import VehicleDetector
from src.utils.config import load_config
from src.utils.model_paths import resolve_yolo_model


class TrafficLightCollector:
    """红绿灯数据采集器"""

    def __init__(
        self,
        video_source: str,
        output_dir: str = "datasets/traffic_light_cls",
        model_path: str = "models/yolo12n_vehicle.pt",
        confidence: float = 0.15,  # 降低阈值，提高红绿灯召回
        device: str = "cuda",
        expand_px: int = 5,  # ROI 外扩像素
    ):
        self.video_source = video_source
        self.output_dir = Path(output_dir)
        self.expand_px = expand_px

        # 创建分类目录
        for cls_name in ["red", "yellow", "green", "off"]:
            (self.output_dir / cls_name).mkdir(parents=True, exist_ok=True)

        # 初始化 YOLO 检测器
        resolved, fallback = resolve_yolo_model(model_path)
        if fallback:
            print(f"[WARN] Using fallback model: {resolved}")
        self.detector = VehicleDetector(
            model_path=resolved,
            confidence=confidence,
            device=device,
            imgsz=640,
            enable_tiling=False,  # 数据采集阶段不切片，保证速度
        )

        # 状态
        self.stats = {"red": 0, "yellow": 0, "green": 0, "off": 0}
        self.light_bbox = None  # 当前帧的红绿灯 bbox
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.manual_bbox = None

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调：手动拖框"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.draw_start = (x, y)
            self.draw_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.draw_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.draw_end = (x, y)
            x1 = min(self.draw_start[0], self.draw_end[0])
            y1 = min(self.draw_start[1], self.draw_end[1])
            x2 = max(self.draw_start[0], self.draw_end[0])
            y2 = max(self.draw_start[1], self.draw_end[1])
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.manual_bbox = (x1, y1, x2, y2)

    def _detect_light(self, frame: np.ndarray):
        """检测红绿灯 bbox"""
        all_dets = self.detector.detect(frame, [self.detector.TRAFFIC_LIGHT_CLASS])
        if all_dets:
            best = max(all_dets, key=lambda d: d.confidence)
            self.light_bbox = best.bbox
            return best.confidence
        self.light_bbox = None
        return 0.0

    def _save_roi(self, frame: np.ndarray, bbox, label: str):
        """保存 ROI 裁剪图"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # 外扩
        x1 = max(0, x1 - self.expand_px)
        y1 = max(0, y1 - self.expand_px)
        x2 = min(w, x2 + self.expand_px)
        y2 = min(h, y2 + self.expand_px)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("  [SKIP] ROI is empty")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"tl_{label}_{timestamp}.jpg"
        filepath = self.output_dir / label / filename
        cv2.imwrite(str(filepath), roi)
        self.stats[label] += 1
        print(f"  [SAVED] {label}: {filepath.name} ({roi.shape[1]}x{roi.shape[0]})")

    def run(self):
        """运行数据采集主循环"""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.video_source}")
            return

        cv2.namedWindow("Traffic Light Collector")
        cv2.setMouseCallback("Traffic Light Collector", self._mouse_callback)

        frame_idx = 0
        show_stats = True
        self.manual_bbox = None

        print("=" * 60)
        print("红绿灯数据采集工具")
        print("=" * 60)
        print("按键: r=红灯 y=黄灯 g=绿灯 o=灭灯 Space=跳过 d=拖框 q=退出 s=隐藏统计 +/-=调外扩")
        print(f"输出目录: {self.output_dir.resolve()}")
        print(f"ROI外扩: {self.expand_px}px")
        print("=" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video ended, exiting.")
                break

            frame_idx += 1
            display = frame.copy()
            h, w = display.shape[:2]

            # 每帧检测红绿灯
            conf = self._detect_light(frame)

            # 绘制检测框
            active_bbox = self.manual_bbox or self.light_bbox
            if active_bbox:
                x1, y1, x2, y2 = active_bbox
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # 外扩预览
                ex1 = max(0, x1 - self.expand_px)
                ey1 = max(0, y1 - self.expand_px)
                ex2 = min(w, x2 + self.expand_px)
                ey2 = min(h, y2 + self.expand_px)
                cv2.rectangle(display, (ex1, ey1), (ex2, ey2), (255, 200, 0), 1)
                cv2.putText(display, f"conf={conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 拖框预览
            if self.drawing and self.draw_start and self.draw_end:
                cv2.rectangle(display, self.draw_start, self.draw_end, (255, 0, 0), 1)

            # 状态栏
            cv2.putText(display, f"Frame: {frame_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display, f"Expand: {self.expand_px}px", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if active_bbox:
                cv2.putText(display, "Press: r/y/g/o to label", (10, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(display, "No TL detected - press 'd' to draw bbox", (10, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # 统计面板
            if show_stats:
                y_off = 80
                for label, count in self.stats.items():
                    colors = {"red": (0, 0, 255), "yellow": (0, 255, 255),
                              "green": (0, 255, 0), "off": (128, 128, 128)}
                    cv2.putText(display, f"{label}: {count}", (10, y_off),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 1)
                    y_off += 20

            cv2.imshow("Traffic Light Collector", display)

            key = cv2.waitKey(0) & 0xFF  # 逐帧等待

            # 处理按键
            label_map = {
                ord('r'): 'red',
                ord('y'): 'yellow',
                ord('g'): 'green',
                ord('o'): 'off',
            }

            if key in label_map:
                bbox = self.manual_bbox or self.light_bbox
                if bbox:
                    self._save_roi(frame, bbox, label_map[key])
                else:
                    print("  [SKIP] No bbox available, press 'd' to draw one first")
                self.manual_bbox = None

            elif key == ord(' '):  # 跳过
                self.manual_bbox = None
                pass

            elif key == ord('d'):  # 手动拖框模式
                print("  [MODE] Draw bbox with mouse (click & drag), then press r/y/g/o to label")
                self.manual_bbox = None  # 等待鼠标事件设置

            elif key == ord('s'):  # 切换统计
                show_stats = not show_stats

            elif key == ord('+') or key == ord('='):
                self.expand_px = min(50, self.expand_px + 2)
                print(f"  Expand: {self.expand_px}px")

            elif key == ord('-'):
                self.expand_px = max(0, self.expand_px - 2)
                print(f"  Expand: {self.expand_px}px")

            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # 打印统计
        print("\n" + "=" * 60)
        print("采集完成! 统计:")
        total = sum(self.stats.values())
        for label, count in self.stats.items():
            print(f"  {label}: {count}")
        print(f"  总计: {total}")
        print(f"  输出目录: {self.output_dir.resolve()}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="红绿灯 ROI 数据采集工具")
    parser.add_argument("--source", "-s", default="0",
                        help="视频源（摄像头ID / RTSP / 视频文件）")
    parser.add_argument("--output", "-o", default="datasets/traffic_light_cls",
                        help="输出目录")
    parser.add_argument("--model", "-m", default="models/yolo12n_vehicle.pt",
                        help="YOLO 模型路径")
    parser.add_argument("--confidence", "-c", type=float, default=0.15,
                        help="红绿灯检测置信度阈值（建议调低）")
    parser.add_argument("--device", "-d", default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--expand", "-e", type=int, default=5,
                        help="ROI 外扩像素")
    args = parser.parse_args()

    collector = TrafficLightCollector(
        video_source=args.source,
        output_dir=args.output,
        model_path=args.model,
        confidence=args.confidence,
        device=args.device,
        expand_px=args.expand,
    )
    collector.run()


if __name__ == "__main__":
    main()
