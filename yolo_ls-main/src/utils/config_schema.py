"""配置 schema 校验与默认值合并。"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

ConfigDict = Dict[str, Any]


class ConfigValidationError(ValueError):
    """配置校验失败。"""


# 与 config/settings.yaml 对齐的默认结构（仅补缺，不覆盖已有键）
DEFAULT_CONFIG: ConfigDict = {
    "system": {"device": "cuda", "log_level": "INFO"},
    "performance": {
        "enabled": True,
        "target_fps": 15,
        "warmup_frames": 45,
        "low_fps_checks": 5,
        "dynamic_resolution": True,
        "frame_skip": 1,
    },
    "video": {"source": "0", "fps": 15, "width": 1280, "height": 720},
    "detector": {
        "confidence": 0.3,
        "iou_threshold": 0.45,
        "imgsz": 768,
        "max_det": 300,
        "enable_tiling": False,
    },
    "tracker": {
        "track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "min_box_area": 10,
    },
    "feature": {"pixel_to_meter": 0.05},
    "violation": {"speed_limit": 60},
    "ocr": {"enabled": True, "interval": 10},
    "database": {"path": "data/traffic.db", "pool_size": 5},
    "risk": {
        "enabled": True,
        "history_length": 10,
        "prediction_horizon": 15,
        "collision_threshold": 150.0,
        "interval": 3,
        "max_tracks": 20,
        "ttc_thresholds": {
            "critical": 0.5,
            "high": 1.0,
            "medium": 2.0,
            "low": 3.0,
        },
    },
}

_KNOWN_TOP_LEVEL = frozenset(DEFAULT_CONFIG) | frozenset(
    {"gui", "violation", "ocr", "database", "risk", "feature", "traffic_light"}
)


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    """递归合并，override 优先。"""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _section(cfg: ConfigDict, name: str, path: str) -> ConfigDict:
    raw = cfg.get(name)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"{path} must be a mapping, got {type(raw).__name__}")
    return raw


def _positive_float(
    value: Any,
    path: str,
    *,
    default: Optional[float] = None,
    allow_zero: bool = False,
) -> float:
    if value is None:
        if default is None:
            raise ConfigValidationError(f"{path} is required")
        return float(default)
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{path} must be a number, got {value!r}") from exc
    if allow_zero:
        if v < 0:
            raise ConfigValidationError(f"{path} must be >= 0, got {v}")
    elif v <= 0:
        raise ConfigValidationError(f"{path} must be > 0, got {v}")
    return v


def _positive_int(value: Any, path: str, *, default: Optional[int] = None) -> int:
    if value is None:
        if default is None:
            raise ConfigValidationError(f"{path} is required")
        return int(default)
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{path} must be an integer, got {value!r}") from exc
    if v <= 0:
        raise ConfigValidationError(f"{path} must be > 0, got {v}")
    return v


def _ratio(value: Any, path: str, *, default: float = 0.5) -> float:
    v = _positive_float(value, path, default=default, allow_zero=True)
    if v > 1.0:
        raise ConfigValidationError(f"{path} must be in [0, 1], got {v}")
    return v


def _one_of(value: Any, path: str, choices: Sequence[str], *, default: str) -> str:
    if value is None:
        return default
    s = str(value).lower()
    if s not in choices:
        raise ConfigValidationError(f"{path} must be one of {choices}, got {value!r}")
    return s


def _validate_ttc_thresholds(raw: Any, path: str) -> Dict[str, float]:
    if raw is None:
        raw = DEFAULT_CONFIG["risk"]["ttc_thresholds"]
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"{path} must be a mapping")
    keys = ("critical", "high", "medium", "low")
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = _positive_float(raw.get(k), f"{path}.{k}")
    if not (out["critical"] <= out["high"] <= out["medium"] <= out["low"]):
        raise ConfigValidationError(
            f"{path} must satisfy critical <= high <= medium <= low, got {out}"
        )
    return out


def _non_negative_int(value: Any, path: str, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{path} must be an integer, got {value!r}") from exc
    if v < 0:
        raise ConfigValidationError(f"{path} must be >= 0, got {v}")
    return v


def _validate_stop_line(raw: Any, path: str) -> Optional[ConfigDict]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigValidationError(f"{path} must be a mapping")
    x_start = _non_negative_int(raw.get("x_start"), f"{path}.x_start", default=0)
    x_end = _non_negative_int(raw.get("x_end"), f"{path}.x_end", default=1)
    if x_end <= x_start:
        raise ConfigValidationError(f"{path}.x_end must be greater than x_start")
    return {
        "y": _non_negative_int(raw.get("y"), f"{path}.y", default=0),
        "x_start": x_start,
        "x_end": x_end,
    }


def validate_config(cfg: Optional[ConfigDict], *, strict_unknown: bool = False) -> ConfigDict:
    """
    校验并规范化配置字典。

    Args:
        cfg: 从 YAML 加载的原始配置
        strict_unknown: 为 True 时，未知顶层键报错；默认仅记录 warning

    Returns:
        合并默认值并规范化类型后的配置

    Raises:
        ConfigValidationError: 关键字段非法
    """
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ConfigValidationError(f"config root must be a mapping, got {type(cfg).__name__}")

    for key in cfg:
        if key not in _KNOWN_TOP_LEVEL and key not in DEFAULT_CONFIG:
            msg = f"Unknown top-level config key: {key!r}"
            if strict_unknown:
                raise ConfigValidationError(msg)
            logger.warning(msg)

    merged = _deep_merge(DEFAULT_CONFIG, cfg)
    out: ConfigDict = {}

    system = _section(merged, "system", "system")
    out["system"] = {
        "device": _one_of(system.get("device"), "system.device", ("cuda", "cpu"), default="cuda"),
        "log_level": str(system.get("log_level", "INFO")).upper(),
    }

    perf = _section(merged, "performance", "performance")
    out["performance"] = {
        "enabled": bool(perf.get("enabled", True)),
        "target_fps": _positive_float(perf.get("target_fps"), "performance.target_fps", default=15),
        "warmup_frames": _positive_int(perf.get("warmup_frames"), "performance.warmup_frames", default=45),
        "low_fps_checks": _positive_int(perf.get("low_fps_checks"), "performance.low_fps_checks", default=5),
        "dynamic_resolution": bool(perf.get("dynamic_resolution", True)),
        "frame_skip": _positive_int(perf.get("frame_skip"), "performance.frame_skip", default=1),
        "onnx_runtime": bool(perf.get("onnx_runtime", False)),
        "tensorrt": bool(perf.get("tensorrt", False)),
    }

    video = _section(merged, "video", "video")
    out["video"] = dict(video)
    out["video"]["fps"] = _positive_float(video.get("fps"), "video.fps", default=15)
    out["video"]["width"] = _positive_int(video.get("width"), "video.width", default=1280)
    out["video"]["height"] = _positive_int(video.get("height"), "video.height", default=720)
    if "source" not in out["video"]:
        out["video"]["source"] = "0"

    det = _section(merged, "detector", "detector")
    out["detector"] = dict(det)
    out["detector"]["confidence"] = _ratio(det.get("confidence"), "detector.confidence", default=0.2)
    out["detector"]["iou_threshold"] = _ratio(
        det.get("iou_threshold"), "detector.iou_threshold", default=0.45
    )
    out["detector"]["imgsz"] = _positive_int(det.get("imgsz"), "detector.imgsz", default=768)
    out["detector"]["max_det"] = _positive_int(det.get("max_det"), "detector.max_det", default=300)
    tiling_overlap = det.get("tiling_overlap", 0.2)
    try:
        ov = float(tiling_overlap)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError("detector.tiling_overlap must be a number") from exc
    if not 0.0 <= ov <= 0.5:
        raise ConfigValidationError(f"detector.tiling_overlap must be in [0, 0.5], got {ov}")
    out["detector"]["tiling_overlap"] = ov

    tracker = _section(merged, "tracker", "tracker")
    out["tracker"] = dict(tracker)
    for key in ("track_thresh", "match_thresh"):
        if key in tracker or key in DEFAULT_CONFIG["tracker"]:
            out["tracker"][key] = _ratio(
                tracker.get(key, DEFAULT_CONFIG["tracker"][key]),
                f"tracker.{key}",
                default=float(DEFAULT_CONFIG["tracker"][key]),
            )
    out["tracker"]["track_buffer"] = _positive_int(
        tracker.get("track_buffer"), "tracker.track_buffer", default=30
    )
    out["tracker"]["min_box_area"] = _positive_int(
        tracker.get("min_box_area"), "tracker.min_box_area", default=10
    )

    feature = _section(merged, "feature", "feature")
    out["feature"] = dict(feature)
    ptm = feature.get("pixel_to_meter", 0.05)
    out["feature"]["pixel_to_meter"] = _positive_float(
        ptm, "feature.pixel_to_meter", default=0.05
    )

    vio = _section(merged, "violation", "violation")
    out["violation"] = dict(vio)
    if "speed_limit" in vio or "speed_limit" in DEFAULT_CONFIG["violation"]:
        out["violation"]["speed_limit"] = _positive_float(
            vio.get("speed_limit"), "violation.speed_limit", default=60
        )
    if "stop_line" in vio:
        out["violation"]["stop_line"] = _validate_stop_line(vio["stop_line"], "violation.stop_line")

    ocr = _section(merged, "ocr", "ocr")
    out["ocr"] = dict(ocr)
    if "interval" in ocr:
        interval = ocr.get("interval")
        if interval is not None:
            out["ocr"]["interval"] = _positive_int(interval, "ocr.interval", default=10)

    db = _section(merged, "database", "database")
    out["database"] = dict(db)
    out["database"]["pool_size"] = _positive_int(db.get("pool_size"), "database.pool_size", default=5)
    if "retention_days" in db:
        out["database"]["retention_days"] = _positive_int(
            db.get("retention_days"), "database.retention_days", default=30
        )

    risk = _section(merged, "risk", "risk")
    out["risk"] = dict(risk)
    out["risk"]["history_length"] = _positive_int(
        risk.get("history_length"), "risk.history_length", default=10
    )
    out["risk"]["prediction_horizon"] = _positive_int(
        risk.get("prediction_horizon"), "risk.prediction_horizon", default=15
    )
    out["risk"]["collision_threshold"] = _positive_float(
        risk.get("collision_threshold"), "risk.collision_threshold", default=150.0
    )
    out["risk"]["interval"] = _positive_int(risk.get("interval"), "risk.interval", default=3)
    out["risk"]["max_tracks"] = _positive_int(risk.get("max_tracks"), "risk.max_tracks", default=20)
    out["risk"]["ttc_thresholds"] = _validate_ttc_thresholds(
        risk.get("ttc_thresholds"), "risk.ttc_thresholds"
    )

    for section in ("gui", "traffic_light"):
        if section in merged:
            sec = merged[section]
            if sec is not None and not isinstance(sec, dict):
                raise ConfigValidationError(f"{section} must be a mapping")
            out[section] = sec

    return _deep_merge(merged, out)
