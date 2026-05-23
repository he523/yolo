"""配置 schema 校验单元测试。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config_schema import ConfigValidationError, validate_config


def test_validate_merges_defaults():
    cfg = validate_config({})
    assert cfg["video"]["fps"] == 15
    assert cfg["detector"]["confidence"] == 0.2


def test_validate_rejects_invalid_confidence():
    with pytest.raises(ConfigValidationError):
        validate_config({"detector": {"confidence": 1.5}})


def test_validate_rejects_zero_fps():
    with pytest.raises(ConfigValidationError):
        validate_config({"video": {"fps": 0}})


def test_validate_ttc_ordering():
    with pytest.raises(ConfigValidationError):
        validate_config({
            "risk": {
                "ttc_thresholds": {
                    "critical": 2.0,
                    "high": 1.0,
                    "medium": 2.0,
                    "low": 3.0,
                }
            }
        })


def test_validate_stop_line():
    cfg = validate_config({
        "violation": {"stop_line": {"y": 100, "x_start": 10, "x_end": 200}}
    })
    assert cfg["violation"]["stop_line"]["y"] == 100


def test_validate_preserves_extra_detector_keys():
    cfg = validate_config({
        "detector": {"model_path": "models/custom.pt", "enable_tiling": True}
    })
    assert cfg["detector"]["model_path"] == "models/custom.pt"
    assert cfg["detector"]["enable_tiling"] is True


def test_load_project_settings_yaml():
    """加载仓库默认 settings.yaml（需安装 pyyaml）。"""
    pytest.importorskip("yaml")
    from src.utils.config import load_config

    cfg = load_config("config/settings.yaml", validate=True)
    assert cfg["video"]["fps"] > 0
    assert 0 < cfg["detector"]["confidence"] <= 1
