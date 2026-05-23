"""Automatic stop line detection from traffic camera frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class StopLineCandidate:
    """A detected horizontal road marking candidate."""

    y_roi: int
    x_start: int
    x_end: int
    run_length: int
    thickness: int
    score: float


def _longest_run(row: np.ndarray) -> Tuple[int, int, int]:
    """Return (start, end_exclusive, length) of the longest 255 run in a 1-D row."""
    if row.size == 0:
        return 0, 0, 0

    padded = np.concatenate(([0], (row > 0).astype(np.uint8), [0]))
    diff = np.diff(padded.astype(np.int32))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0, 0, 0

    lengths = ends - starts
    idx = int(np.argmax(lengths))
    return int(starts[idx]), int(ends[idx]), int(lengths[idx])


def _build_marking_mask(roi: np.ndarray) -> np.ndarray:
    """Extract bright road markings (white/yellow) on asphalt."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
    yellow = cv2.inRange(hsv, (12, 70, 110), (40, 255, 255))

    # Top-hat highlights thin bright structures on dark pavement.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, bright = cv2.threshold(tophat, 25, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(white, yellow)
    mask = cv2.bitwise_or(mask, bright)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -8,
    )
    mask = cv2.bitwise_and(mask, adaptive)

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(int(roi.shape[1] * 0.12), 15), 3),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _estimate_thickness(mask: np.ndarray, y: int, x_start: int, x_end: int) -> int:
    """Count consecutive rows around y that still contain the same horizontal span."""
    h = mask.shape[0]
    thickness = 0
    half_span = max((x_end - x_start) // 3, 20)
    xs = max(x_start, 0)
    xe = min(x_end, mask.shape[1])

    for dy in range(-6, 7):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        row = mask[yy, xs:xe]
        if row.size == 0:
            continue
        _, _, run_len = _longest_run(row)
        if run_len >= half_span:
            thickness += 1
    return max(thickness, 1)


def _build_run_profile(mask: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Return smoothed horizontal run-length profile and per-row spans."""
    h = mask.shape[0]
    profile = np.zeros(h, dtype=np.float32)
    spans: List[Tuple[int, int, int]] = []

    for y in range(h):
        run_start, run_end, run_len = _longest_run(mask[y])
        profile[y] = float(run_len)
        spans.append((run_start, run_end, run_len))

    kernel = np.ones(9, dtype=np.float32) / 9.0
    smooth = np.convolve(profile, kernel, mode="same")
    return smooth, spans


def _find_line_peaks(
    mask: np.ndarray,
    min_run_ratio: float = 0.22,
) -> List[StopLineCandidate]:
    """Find local maxima in the horizontal run-length profile."""
    h, w = mask.shape[:2]
    min_run = max(int(w * min_run_ratio), 40)
    smooth, spans = _build_run_profile(mask)

    raw_peaks: List[StopLineCandidate] = []
    for y in range(2, h - 2):
        value = smooth[y]
        if value < min_run:
            continue
        if not (value >= smooth[y - 1] and value >= smooth[y + 1]):
            continue
        if not (value >= smooth[y - 2] and value >= smooth[y + 2]):
            continue

        run_start, run_end, run_len = spans[y]
        if run_len < min_run:
            continue

        thickness = _estimate_thickness(mask, y, run_start, run_end)
        raw_peaks.append(
            StopLineCandidate(
                y_roi=y,
                x_start=run_start,
                x_end=run_end - 1,
                run_length=run_len,
                thickness=thickness,
                score=float(value) + thickness * 12.0,
            )
        )

    if not raw_peaks:
        return []

    ordered = sorted(raw_peaks, key=lambda c: c.y_roi)
    merged: List[StopLineCandidate] = [ordered[0]]
    merge_gap = 14

    for peak in ordered[1:]:
        prev = merged[-1]
        if peak.y_roi - prev.y_roi <= merge_gap:
            if peak.score >= prev.score:
                merged[-1] = peak
        else:
            merged.append(peak)

    return merged


def _crosswalk_penalty(
    peaks: List[StopLineCandidate],
    target: StopLineCandidate,
    band: int = 28,
) -> float:
    """
    Penalize peaks sandwiched between similar-width stripes (zebra crossing).
    """
    above = [
        peak for peak in peaks
        if target.y_roi - band <= peak.y_roi < target.y_roi - 4
    ]
    below = [
        peak for peak in peaks
        if target.y_roi + 4 < peak.y_roi <= target.y_roi + band
    ]
    if not above or not below:
        return 0.0

    similar_above = sum(
        1 for peak in above
        if peak.run_length >= target.run_length * 0.55
    )
    similar_below = sum(
        1 for peak in below
        if peak.run_length >= target.run_length * 0.55
    )
    if similar_above >= 1 and similar_below >= 1:
        return float(target.run_length * 0.65)
    return 0.0


def _position_score(y_global: int, frame_h: int) -> float:
    """Prefer stop lines in the lower-middle part of the frame."""
    ratio = y_global / max(frame_h - 1, 1)
    # Typical intersection cameras place the stop line around 44%-56% height.
    center = 0.50
    width = 0.12
    distance = abs(ratio - center) / width
    return max(0.0, 1.0 - distance)


def _score_candidate(
    candidate: StopLineCandidate,
    roi_top: int,
    frame_h: int,
    frame_w: int,
    max_run: int,
    crosswalk_penalty: float = 0.0,
) -> float:
    y_global = roi_top + candidate.y_roi
    run_ratio = candidate.run_length / max(max_run, 1)
    width_ratio = (candidate.x_end - candidate.x_start + 1) / max(frame_w, 1)

    score = (
        run_ratio * frame_w * 1.45
        + candidate.thickness * 14.0
        + _position_score(y_global, frame_h) * frame_w * 0.85
        + width_ratio * frame_w * 0.20
        - crosswalk_penalty
    )

    ratio = y_global / max(frame_h - 1, 1)
    if ratio >= 0.66:
        score -= frame_w * 1.4
    if ratio <= 0.34:
        score -= frame_w * 0.6
    if width_ratio < 0.28:
        score -= frame_w * 0.35
    return score


def _select_stop_line_peak(
    peaks: List[StopLineCandidate],
    roi_top: int,
    frame_h: int,
    frame_w: int,
    mask_h: int,
) -> Optional[StopLineCandidate]:
    """Pick the most likely stop line among detected peaks."""
    if not peaks:
        return None

    max_run = max(peak.run_length for peak in peaks)
    min_run = max(int(max_run * 0.68), int(frame_w * 0.28))

    candidates = [
        peak for peak in peaks
        if peak.y_roi <= int(mask_h * 0.88)
        and peak.run_length >= min_run
    ]
    if not candidates:
        return None

    preferred = [
        peak for peak in candidates
        if 0.40 <= (roi_top + peak.y_roi) / max(frame_h - 1, 1) <= 0.60
    ]
    pool = preferred or candidates

    scored = [
        (
            peak,
            _score_candidate(
                peak,
                roi_top,
                frame_h,
                frame_w,
                max_run,
                _crosswalk_penalty(peaks, peak),
            ),
        )
        for peak in pool
    ]
    return max(scored, key=lambda item: item[1])[0]


def _hough_refinement(
    mask: np.ndarray,
    candidate: StopLineCandidate,
    search_band: int = 12,
) -> StopLineCandidate:
    """Refine y/x span using probabilistic Hough near the chosen row."""
    h, w = mask.shape[:2]
    y0 = max(candidate.y_roi - search_band, 0)
    y1 = min(candidate.y_roi + search_band + 1, h)
    band = mask[y0:y1]

    lines = cv2.HoughLinesP(
        band,
        rho=1,
        theta=np.pi / 180,
        threshold=max(30, int(w * 0.08)),
        minLineLength=max(int(w * 0.18), 60),
        maxLineGap=max(int(w * 0.04), 25),
    )
    if lines is None:
        return candidate

    xs: List[int] = []
    ys: List[float] = []
    for line in lines:
        x1, y1_l, x2, y2_l = line[0]
        dx = x2 - x1
        dy = y2_l - y1_l
        if dx == 0:
            continue
        if abs(dy / dx) > 0.12:
            continue
        xs.extend([x1, x2])
        ys.append((y1_l + y2_l) / 2.0 + y0)

    if not xs or not ys:
        return candidate

    return StopLineCandidate(
        y_roi=int(round(float(np.median(ys)))),
        x_start=max(0, int(min(xs))),
        x_end=min(w - 1, int(max(xs))),
        run_length=candidate.run_length,
        thickness=candidate.thickness,
        score=candidate.score,
    )


def detect_stop_line(
    frame: np.ndarray,
    roi_top_ratio: float = 0.30,
) -> Optional[Tuple[int, int, int]]:
    """
    Detect a stop line in a BGR frame.

    Args:
        frame: Full-color BGR image.
        roi_top_ratio: Start scanning from this fraction of frame height.

    Returns:
        ``(y, x_start, x_end)`` in full-frame coordinates, or ``None``.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    if h < 40 or w < 80:
        return None

    roi_top = int(h * roi_top_ratio)
    roi = frame[roi_top:h, 0:w]
    mask = _build_marking_mask(roi)

    peaks = _find_line_peaks(mask)
    if not peaks:
        return None

    best = _select_stop_line_peak(peaks, roi_top, h, w, mask.shape[0])
    if best is None:
        return None

    refined = _hough_refinement(mask, best)
    y_global = roi_top + refined.y_roi
    return y_global, refined.x_start, refined.x_end
