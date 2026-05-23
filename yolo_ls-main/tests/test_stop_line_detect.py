"""Tests for automatic stop line detection."""
from pathlib import Path

import cv2
import pytest

from src.utils.stop_line_detect import detect_stop_line

SNAPSHOT = (
    Path(__file__).resolve().parents[1]
    / "data/snapshots/violations/20260523_182548_042214_speeding.jpg"
)


@pytest.mark.skipif(not SNAPSHOT.exists(), reason="sample snapshot not available")
def test_detect_stop_line_on_traffic_snapshot():
    frame = cv2.imread(str(SNAPSHOT))
    assert frame is not None

    h, w = frame.shape[:2]
    result = detect_stop_line(frame)
    assert result is not None

    y, x_start, x_end = result
    # Stop line sits near the middle-lower part of typical intersection views.
    assert int(h * 0.40) <= y <= int(h * 0.60)
    assert x_end - x_start >= int(w * 0.45)
    assert 0 <= x_start < x_end < w


def test_detect_stop_line_returns_none_for_blank_frame():
    frame = 255 * __import__("numpy").ones((240, 320, 3), dtype="uint8")
    assert detect_stop_line(frame) is None
