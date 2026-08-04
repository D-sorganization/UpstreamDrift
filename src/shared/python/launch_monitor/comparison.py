"""Matched and descriptive cross-monitor comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonitorSummary:
    monitor: str
    sample_count: int
    mean: float
    standard_deviation: float
    median: float


@dataclass(frozen=True)
class PairwiseMonitorComparison:
    reference: str
    comparator: str
    matched: bool
    sample_count: int
    mean_bias: float
    standard_deviation_bias: float
    lower_limit: float
    upper_limit: float
    slope: float
    intercept: float
    correlation: float
    warning: str | None


@dataclass(frozen=True)
class MonitorComparisonResult:
    metric: str
    summaries: tuple[MonitorSummary, ...]
    pairwise: tuple[PairwiseMonitorComparison, ...]


def compare_monitors(
    frame: pd.DataFrame,
    *,
    metric: str,
    monitor_column: str = "monitor_vendor",
    match_column: str | None = None,
    reference_monitor: str | None = None,
) -> MonitorComparisonResult:
    """Compare monitor distributions or matched-shot measurement behavior."""
    required = {metric, monitor_column}
    if match_column:
        required.add(match_column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Columns not present: {sorted(missing)}")
    clean = frame[list(required)].copy()
    clean[metric] = pd.to_numeric(clean[metric], errors="coerce")
    clean = clean.dropna(subset=[metric, monitor_column])
    monitors = sorted(clean[monitor_column].astype(str).unique())
    if len(monitors) < 2:
        raise ValueError("At least two monitors are required")
    reference = reference_monitor or monitors[0]
    if reference not in monitors:
        raise ValueError(f"Reference monitor not present: {reference}")
    summaries = tuple(
        MonitorSummary(
            monitor,
            int(len(values)),
            float(values.mean()),
            float(values.std(ddof=1)),
            float(values.median()),
        )
        for monitor in monitors
        for values in [clean.loc[clean[monitor_column].astype(str) == monitor, metric]]
    )
    pairwise: list[PairwiseMonitorComparison] = []
    for comparator in monitors:
        if comparator == reference:
            continue
        if match_column:
            selected = clean[
                clean[monitor_column].astype(str).isin([reference, comparator])
            ]
            pivot = selected.pivot_table(
                index=match_column,
                columns=monitor_column,
                values=metric,
                aggfunc="mean",
            ).dropna(subset=[reference, comparator])
            if len(pivot) < 3:
                raise ValueError(
                    f"At least three matched pairs are required for {reference} vs {comparator}"
                )
            x = pivot[reference].to_numpy(float)
            y = pivot[comparator].to_numpy(float)
            difference = y - x
            slope, intercept = np.polyfit(x, y, 1)
            bias = float(difference.mean())
            std_bias = float(difference.std(ddof=1))
            pairwise.append(
                PairwiseMonitorComparison(
                    reference,
                    comparator,
                    True,
                    len(pivot),
                    bias,
                    std_bias,
                    bias - 1.96 * std_bias,
                    bias + 1.96 * std_bias,
                    float(slope),
                    float(intercept),
                    float(np.corrcoef(x, y)[0, 1]),
                    None,
                )
            )
        else:
            reference_values = clean.loc[
                clean[monitor_column].astype(str) == reference, metric
            ].to_numpy(float)
            comparator_values = clean.loc[
                clean[monitor_column].astype(str) == comparator, metric
            ].to_numpy(float)
            bias = float(comparator_values.mean() - reference_values.mean())
            pooled = np.sqrt(
                (reference_values.var(ddof=1) + comparator_values.var(ddof=1)) / 2
            )
            effect = bias / pooled if pooled > 0 else float("nan")
            pairwise.append(
                PairwiseMonitorComparison(
                    reference,
                    comparator,
                    False,
                    min(len(reference_values), len(comparator_values)),
                    bias,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float(effect),
                    "Unmatched comparison is descriptive and may be confounded by "
                    "player, club, environment, and session composition.",
                )
            )
    return MonitorComparisonResult(metric, summaries, tuple(pairwise))
