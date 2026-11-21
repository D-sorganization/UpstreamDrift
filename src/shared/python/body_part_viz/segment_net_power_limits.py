"""Robust symmetric limits for segment net-power shading (#15399)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_EXCLUDE_FRAME_RANGES",
    "LimitDerivation",
    "derive_symmetric_limit_w",
]

DEFAULT_EXCLUDE_FRAME_RANGES: tuple[tuple[int, int], ...] = ((661, 664),)
DEFAULT_TRIM_FRAMES = 15
DEFAULT_FILTER_HZ = 12.0
DEFAULT_PERCENTILE = 98.0
DEFAULT_MAX_PHYSICAL_W = 25_000.0


@dataclass(frozen=True)
class LimitDerivation:
    symmetric_limit_w: float
    source: str
    sample_count: int
    raw_peak_w: float
    metadata: dict[str, Any]


def _in_excluded(index: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(lo <= index <= hi for lo, hi in ranges)


def _zero_phase_lowpass(
    q: np.ndarray, *, sample_hz: float, cutoff_hz: float
) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    if q.shape[0] < 8:
        return q.astype(np.float64, copy=False)
    nyquist = 0.5 * sample_hz
    normal = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(2, normal, btype="low")
    padlen = min(3 * (max(len(a), len(b)) - 1), q.shape[0] - 1)
    return np.asarray(filtfilt(b, a, q, axis=0, padlen=padlen), dtype=np.float64)


def _despike_median(q: np.ndarray, *, window: int = 5) -> np.ndarray:
    if q.shape[0] < window:
        return q
    half = window // 2
    out = q.copy()
    for axis in range(q.shape[1]):
        column = q[:, axis]
        for idx in range(half, len(column) - half):
            window_vals = column[idx - half : idx + half + 1]
            med = float(np.median(window_vals))
            if abs(column[idx] - med) > 6.0 * float(np.std(window_vals) + 1e-9):
                out[idx, axis] = med
    return out


def derive_symmetric_limit_w(
    rates_w: Iterable[float],
    *,
    percentile: float = DEFAULT_PERCENTILE,
    max_physical_w: float = DEFAULT_MAX_PHYSICAL_W,
    source: str,
    sample_count: int,
) -> LimitDerivation:
    samples = [float(abs(v)) for v in rates_w if np.isfinite(v)]
    if not samples:
        raise ValueError("rates_w must contain at least one finite sample")
    raw_peak = float(max(samples))
    cap = min(raw_peak, float(max_physical_w))
    limit = float(np.percentile(samples, percentile))
    limit = min(limit, cap)
    if limit <= 0.0:
        limit = max(cap, 1.0)
    return LimitDerivation(
        symmetric_limit_w=limit,
        source=source,
        sample_count=sample_count,
        raw_peak_w=raw_peak,
        metadata={
            "percentile": percentile,
            "max_physical_w": max_physical_w,
        },
    )


def trajectory_mechanical_rates(
    sample_hz: float,
    q_traj: np.ndarray,
    frame_rates: Callable[[np.ndarray, np.ndarray, np.ndarray], Iterable[float]],
    *,
    exclude_ranges: Sequence[tuple[int, int]] = DEFAULT_EXCLUDE_FRAME_RANGES,
    trim_frames: int = DEFAULT_TRIM_FRAMES,
    filter_hz: float = DEFAULT_FILTER_HZ,
) -> tuple[list[float], int]:
    q = _despike_median(np.asarray(q_traj, dtype=np.float64))
    q = _zero_phase_lowpass(q, sample_hz=sample_hz, cutoff_hz=filter_hz)
    dt = 1.0 / sample_hz
    qvel = np.gradient(q, dt, axis=0)
    qacc = np.gradient(qvel, dt, axis=0)
    n = q.shape[0]
    start = trim_frames
    end = n - trim_frames
    if end <= start:
        raise ValueError("trim_frames leaves no samples")
    collected: list[float] = []
    for idx in range(start, end):
        if _in_excluded(idx, exclude_ranges):
            continue
        for rate in frame_rates(q[idx], qvel[idx], qacc[idx]):
            if np.isfinite(rate):
                collected.append(float(rate))
    return collected, max(0, end - start)


def write_limits_json(path: Path, derivation: LimitDerivation, **extra: Any) -> None:
    payload = {
        "symmetric_limit_w": derivation.symmetric_limit_w,
        "source": derivation.source,
        "sample_count": derivation.sample_count,
        "raw_peak_w": derivation.raw_peak_w,
        **derivation.metadata,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
