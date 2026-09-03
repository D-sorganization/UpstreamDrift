"""Longitudinal player-performance trend analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ChangeCandidate:
    """Candidate step change ranked by standardized before/after difference."""

    captured_at: pd.Timestamp
    row_index: int
    before_mean: float
    after_mean: float
    effect_size: float


@dataclass(frozen=True)
class TrendResult:
    """Linear/robust trend, rolling series, EWMA, and change candidates."""

    metric: str
    sample_count: int
    slope_per_day: float
    robust_slope_per_day: float
    p_value: float
    earliest_mean: float
    latest_mean: float
    rolling: pd.DataFrame
    change_candidates: tuple[ChangeCandidate, ...]


def _change_candidates(
    times: pd.Series, values: np.ndarray, minimum_segment: int
) -> tuple[ChangeCandidate, ...]:
    candidates: list[ChangeCandidate] = []
    for split in range(minimum_segment, len(values) - minimum_segment + 1):
        before = values[:split]
        after = values[split:]
        pooled = np.sqrt((before.var(ddof=1) + after.var(ddof=1)) / 2)
        effect = (after.mean() - before.mean()) / pooled if pooled > 0 else 0.0
        candidates.append(
            ChangeCandidate(
                pd.Timestamp(times.iloc[split]),
                split,
                float(before.mean()),
                float(after.mean()),
                float(effect),
            )
        )
    candidates.sort(key=lambda item: -abs(item.effect_size))
    return tuple(item for item in candidates[:3] if abs(item.effect_size) >= 0.5)


def analyze_trend(
    frame: pd.DataFrame,
    *,
    metric: str,
    time_column: str = "captured_at",
    rolling_window: int = 10,
    ewma_span: int | None = None,
) -> TrendResult:
    """Analyze center/variance movement over irregular observation time."""
    if rolling_window < 3:
        raise ValueError("rolling_window must be at least three")
    missing = {metric, time_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Trend columns not present: {sorted(missing)}")
    clean = frame[[time_column, metric]].copy()
    clean[time_column] = pd.to_datetime(clean[time_column], errors="coerce", utc=True)
    clean[metric] = pd.to_numeric(clean[metric], errors="coerce")
    clean = clean.dropna().sort_values(time_column).reset_index(drop=True)
    if len(clean) < max(6, rolling_window):
        raise ValueError("Insufficient complete observations for trend analysis")
    values = clean[metric].to_numpy(float)
    elapsed_days = (
        clean[time_column] - clean[time_column].iloc[0]
    ).dt.total_seconds().to_numpy() / 86400.0
    regression = stats.linregress(elapsed_days, values)
    robust = stats.theilslopes(values, elapsed_days)
    span = ewma_span or rolling_window
    rolling = pd.DataFrame(
        {
            time_column: clean[time_column],
            "value": values,
            "rolling_mean": clean[metric].rolling(rolling_window, min_periods=3).mean(),
            "rolling_median": clean[metric]
            .rolling(rolling_window, min_periods=3)
            .median(),
            "rolling_std": clean[metric].rolling(rolling_window, min_periods=3).std(),
            "ewma": clean[metric].ewm(span=span, adjust=False).mean(),
        }
    )
    segment = max(3, min(rolling_window, len(values) // 3))
    return TrendResult(
        metric,
        len(clean),
        float(regression.slope),
        float(robust.slope),
        float(regression.pvalue),
        float(np.mean(values[:segment])),
        float(np.mean(values[-segment:])),
        rolling,
        _change_candidates(clean[time_column], values, segment),
    )
