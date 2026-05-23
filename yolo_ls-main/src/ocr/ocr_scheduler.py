"""OCR 调度：优先识别尚无车牌的轨迹，减少无效重复识别。"""
import re
from typing import Dict, List, Optional, Tuple


class OCRScheduler:
    """
    每帧从候选轨迹中选出最多 N 个进行 OCR。
    优先级：无缓存 > 缓存为「识别中」> 已有车牌（跳过）。
    """

    def __init__(self, max_per_frame: int = 8):
        self.max_per_frame = max(1, int(max_per_frame))

    @staticmethod
    def _needs_ocr(cached: Optional[str]) -> bool:
        if cached is None:
            return True
        if cached == "识别中":
            return True
        if len(cached) < 5:
            return True
        if not re.search(r'[A-Z0-9]', cached.upper()):
            return True
        return False

    def select_tracks(
        self,
        track_ids: List[int],
        plate_cache: Dict[int, str],
        bbox_heights: Dict[int, int],
        min_bbox_height: int,
    ) -> List[int]:
        """返回本帧应执行 OCR 的 track_id 列表。"""
        pending: List[int] = []

        for tid in track_ids:
            h = bbox_heights.get(tid, 0)
            if h < min_bbox_height:
                continue
            if self._needs_ocr(plate_cache.get(tid)):
                pending.append(tid)

        return pending[: self.max_per_frame]
