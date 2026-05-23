"""VideoStream 行为测试。"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.video.stream import VideoStream


def test_len_requires_open():
    stream = VideoStream(source="0")
    with pytest.raises(RuntimeError):
        len(stream)


def test_len_live_source_raises_type_error():
    stream = VideoStream(source="0")
    stream.cap = MagicMock()
    stream.is_file = False
    with pytest.raises(TypeError):
        len(stream)


def test_bool_live_source_does_not_call_len():
    stream = VideoStream(source="0")
    stream.cap = MagicMock()
    stream.is_file = False
    assert bool(stream) is True


def test_bool_before_open_is_false():
    stream = VideoStream(source="0")
    assert bool(stream) is False


def test_len_file_source():
    stream = VideoStream(source="dummy.mp4")
    stream.cap = MagicMock()
    stream.is_file = True
    stream.cap.get.return_value = 120
    assert len(stream) == 120


def test_get_fps_fallback_when_invalid():
    stream = VideoStream(source="0", fps=25)
    stream.cap = MagicMock()
    stream.cap.get.return_value = 0.0
    assert stream.get_fps() == 25.0
