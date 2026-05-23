"""性能监控与自适应降级（FPS 监控、动态分辨率、跳帧）。"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class DegradationPlan:
    """降级策略输出。"""
    level: int = 0
    frame_skip: int = 1
    imgsz_scale: float = 1.0
    enable_tiling: bool = True
    risk_interval_add: int = 0
    ocr_interval_mul: int = 1
    disable_risk: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level,
            'frame_skip': self.frame_skip,
            'imgsz_scale': self.imgsz_scale,
            'enable_tiling': self.enable_tiling,
            'risk_interval_add': self.risk_interval_add,
            'ocr_interval_mul': self.ocr_interval_mul,
            'disable_risk': self.disable_risk,
        }


class FPSMonitor:
    """滑动窗口 FPS 监控，低于目标时触发降级。"""

    def __init__(
        self,
        target_fps: float = 15.0,
        window_size: int = 30,
        warmup_frames: int = 30,
        low_fps_checks: int = 5,
    ):
        self.target_fps = max(1.0, float(target_fps))
        self.frame_times: Deque[float] = deque(maxlen=max(10, window_size))
        self.warmup_frames = max(0, int(warmup_frames))
        self.low_fps_checks = max(1, int(low_fps_checks))
        self._current_level = 0
        self._frame_count = 0
        self._low_fps_streak = 0

    def tick(self, elapsed_sec: float) -> None:
        self._frame_count += 1
        if elapsed_sec > 0:
            self.frame_times.append(elapsed_sec)

    @property
    def avg_fps(self) -> float:
        if not self.frame_times:
            return self.target_fps
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def check_performance(self) -> Optional[DegradationPlan]:
        if self._frame_count < self.warmup_frames:
            return None
        if len(self.frame_times) < 10:
            return None
        avg_fps = self.avg_fps
        threshold = self.target_fps * 0.8
        if avg_fps >= threshold:
            self._low_fps_streak = 0
            if self._current_level > 0:
                self._current_level -= 1
                logger.info("Performance recovered (%.1f FPS), degradation level -> %d",
                            avg_fps, self._current_level)
            return DegradationPlan(level=self._current_level) if self._current_level else None

        self._low_fps_streak += 1
        if self._low_fps_streak >= self.low_fps_checks and self._current_level < 3:
            self._current_level += 1
            self._low_fps_streak = 0
            logger.warning(
                "Low FPS %.1f < %.1f, degradation level -> %d",
                avg_fps, threshold, self._current_level,
            )
        return self._trigger_degradation(self._current_level)

    @staticmethod
    def _trigger_degradation(level: int) -> DegradationPlan:
        plans = {
            1: DegradationPlan(level=1, imgsz_scale=0.85, risk_interval_add=1, ocr_interval_mul=2),
            2: DegradationPlan(
                level=2, frame_skip=2, imgsz_scale=0.72,
                enable_tiling=False, risk_interval_add=2, ocr_interval_mul=2,
            ),
            3: DegradationPlan(
                level=3, frame_skip=3, imgsz_scale=0.6,
                enable_tiling=False, risk_interval_add=3,
                ocr_interval_mul=3, disable_risk=True,
            ),
        }
        return plans.get(level, DegradationPlan(level=level))


class PerformanceOptimizer:
    """
    根据降级计划调整推理参数。
    支持：动态 imgsz、跳帧、关闭切片、拉长 OCR/风险间隔。
    """

    def __init__(
        self,
        base_imgsz: int = 768,
        dynamic_resolution: bool = True,
        frame_skip: int = 1,
        base_enable_tiling: bool = False,
        onnx_runtime: bool = False,
        tensorrt: bool = False,
    ):
        self.base_imgsz = int(base_imgsz)
        self.dynamic_resolution = bool(dynamic_resolution)
        self.frame_skip = max(1, int(frame_skip))
        self.base_enable_tiling = bool(base_enable_tiling)
        self.onnx_runtime = bool(onnx_runtime)
        self.tensorrt = bool(tensorrt)
        self._plan = DegradationPlan(
            frame_skip=self.frame_skip,
            enable_tiling=self.base_enable_tiling,
        )

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], base_imgsz: int = 768) -> 'PerformanceOptimizer':
        det = cfg or {}
        return cls(
            base_imgsz=base_imgsz,
            dynamic_resolution=det.get('dynamic_resolution', True),
            frame_skip=det.get('frame_skip', 1),
            base_enable_tiling=det.get('enable_tiling', False),
            onnx_runtime=det.get('onnx_runtime', False),
            tensorrt=det.get('tensorrt', False),
        )

    def apply_degradation(self, plan: Optional[DegradationPlan]) -> None:
        if plan is None:
            return
        if plan.level == 0:
            self._plan = DegradationPlan(
                frame_skip=self.frame_skip,
                enable_tiling=self.base_enable_tiling,
            )
        else:
            self._plan = plan

    def should_process_frame(self, frame_index: int) -> bool:
        skip = max(1, self._plan.frame_skip)
        return (frame_index % skip) == 0

    def get_imgsz(self) -> int:
        if not self.dynamic_resolution:
            return self.base_imgsz
        scale = max(0.5, min(1.0, self._plan.imgsz_scale))
        size = int(self.base_imgsz * scale)
        return max(320, (size // 32) * 32)

    def get_enable_tiling(self) -> bool:
        return self._plan.enable_tiling and self.base_enable_tiling

    def get_risk_interval(self, base_interval: int) -> int:
        if self._plan.disable_risk:
            return 10 ** 9
        return base_interval + self._plan.risk_interval_add

    def risk_disabled(self) -> bool:
        return self._plan.disable_risk

    def get_ocr_interval(self, base_interval: int) -> int:
        mul = max(1, self._plan.ocr_interval_mul)
        return max(1, base_interval * mul)

    def get_status(self) -> Dict[str, Any]:
        return {
            'imgsz': self.get_imgsz(),
            'frame_skip': self._plan.frame_skip,
            'degradation_level': self._plan.level,
            'onnx_runtime': self.onnx_runtime,
            'tensorrt': self.tensorrt,
        }
