"""Data-quality report for a drift-anchored kinematic match.

Turns a :class:`DriftAnchoredMatchResult` into a reviewable summary of where
matching is not working and the data are likely noisy or obscured.

A verdict from this module is a *software* verdict on one match run. It is NOT
a capture qualification and NOT an engine qualification (epic #11421 rules):
the default thresholds are provisional software thresholds and must not be
quoted as evidence that a capture or a physics engine is validated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.estimation.drift_anchored_matcher import (
    DriftAnchoredMatchResult,
    SampleLabel,
    WindowRecord,
)

__all__ = [
    "MatchQualityReport",
    "QualityThresholds",
    "SuspectInterval",
    "report_to_dict",
    "summarize_match_quality",
]

_ACCEPTED = SampleLabel.ACCEPTED.value
_OUTLIER = SampleLabel.OUTLIER.value
_UNEXPLAINED = SampleLabel.UNEXPLAINED.value
_GAP = SampleLabel.GAP_FILLED.value
_LABELS = (_ACCEPTED, _OUTLIER, _UNEXPLAINED, _GAP)
_UNUSABLE_WINDOW_FAILURE = 0.5  # window failure fraction above which data are unusable


@dataclass(frozen=True)
class SuspectInterval:
    """Maximal run of identical non-accepted labels.

    Attributes:
        start: First sample index (inclusive).
        stop: One past the last sample index (half-open range).
        label: ``SampleLabel`` value: outlier, unexplained or gap_filled.
        t_start: ``t[start]`` [s].
        t_stop: ``t[stop - 1]`` [s].
        peak_ztcf_divergence: Max ZTCF divergence over observed samples in the
            interval (same units as the result field); NaN if none observed.
        mean_tau_disagreement: Mean of ``tau_disagreement`` over the interval
            and all channels [N*m].
    """

    start: int
    stop: int
    label: str
    t_start: float
    t_stop: float
    peak_ztcf_divergence: float
    mean_tau_disagreement: float


@dataclass(frozen=True)
class MatchQualityReport:
    """Reviewable data-quality summary. Software verdict only, see module doc.

    Fractions are of all samples. RMS values are in observation-sigma units
    over ACCEPTED samples. Per-channel arrays have one entry per torque channel.
    """

    n_samples: int
    accepted_fraction: float
    outlier_fraction: float
    unexplained_fraction: float
    gap_fraction: float
    suspect_intervals: tuple[SuspectInterval, ...]
    longest_unexplained_run: int
    longest_gap_run: int
    replay_rms: float
    estimate_rms: float
    tau_disagreement_p95: np.ndarray
    relative_torque_uncertainty_p95: np.ndarray
    window_failure_fraction: float
    saturated_window_fraction: float
    rate_limited_window_fraction: float
    mean_drift_dominance: float
    verdict: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class QualityThresholds:
    """PROVISIONAL software thresholds (not capture or engine qualification)."""

    max_unexplained_fraction: float = 0.05
    max_gap_fraction: float = 0.2
    max_replay_rms: float = 5.0
    max_window_failure_fraction: float = 0.1
    max_saturated_window_fraction: float = 0.2
    unusable_unexplained_fraction: float = 0.25


def _validate_thresholds(th: QualityThresholds) -> None:
    fractions = {
        "max_unexplained_fraction": th.max_unexplained_fraction,
        "max_gap_fraction": th.max_gap_fraction,
        "max_window_failure_fraction": th.max_window_failure_fraction,
        "max_saturated_window_fraction": th.max_saturated_window_fraction,
        "unusable_unexplained_fraction": th.unusable_unexplained_fraction,
    }
    for name, value in fractions.items():
        require(0.0 <= value <= 1.0, f"{name} must be in [0, 1]", value)
    require(th.max_replay_rms > 0.0, "max_replay_rms must be > 0", th.max_replay_rms)


def _validate_result(res: DriftAnchoredMatchResult) -> int:
    n = int(np.asarray(res.t).size)
    require(n > 0, "match result must contain at least one sample")
    per_sample = {
        "labels": res.labels,
        "tau": res.tau,
        "tau_std": res.tau_std,
        "tau_disagreement": res.tau_disagreement,
        "normalized_residual": res.normalized_residual,
        "replay_residual": res.replay_residual,
        "ztcf_divergence": res.ztcf_divergence,
        "drift_dominance": res.drift_dominance,
    }
    for name, arr in per_sample.items():
        require(
            np.asarray(arr).shape[0] == n,
            f"{name} must have one row per sample ({n})",
            np.asarray(arr).shape,
        )
    m = np.asarray(res.tau).shape[1]
    for name in ("tau_std", "tau_disagreement"):
        require(
            np.asarray(per_sample[name]).shape == (n, m),
            f"{name} must have shape ({n}, {m})",
        )
    return n


def _label_values(res: DriftAnchoredMatchResult) -> np.ndarray:
    return np.array([str(getattr(x, "value", x)) for x in res.labels], dtype=object)


def _runs(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Maximal runs of identical non-accepted labels as (start, stop, label)."""
    out: list[tuple[int, int, str]] = []
    start = 0
    for k in range(1, labels.size + 1):
        if k == labels.size or labels[k] != labels[start]:
            if labels[start] != _ACCEPTED:
                out.append((start, k, str(labels[start])))
            start = k
    return out


def _interval(
    res: DriftAnchoredMatchResult, run: tuple[int, int, str]
) -> SuspectInterval:
    start, stop, label = run
    div = np.asarray(res.ztcf_divergence, dtype=float)[start:stop]
    finite = div[np.isfinite(div)]
    return SuspectInterval(
        start=start,
        stop=stop,
        label=label,
        t_start=float(res.t[start]),
        t_stop=float(res.t[stop - 1]),
        peak_ztcf_divergence=float(finite.max()) if finite.size else float("nan"),
        mean_tau_disagreement=float(np.mean(res.tau_disagreement[start:stop])),
    )


def _rms_over(values: np.ndarray, keep: np.ndarray) -> float:
    sel = np.asarray(values, dtype=float)[keep]
    sel = sel[np.isfinite(sel)]
    return float(np.sqrt(np.mean(sel**2))) if sel.size else float("nan")


def _window_fraction(
    windows: tuple[WindowRecord, ...], pred: Callable[[WindowRecord], bool]
) -> float:
    return sum(1 for w in windows if pred(w)) / len(windows) if windows else 0.0


def _longest(runs: list[tuple[int, int, str]], label: str) -> int:
    return max((b - a for a, b, lab in runs if lab == label), default=0)


def _verdict(
    rep: dict[str, float], th: QualityThresholds
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    checks = (
        ("unexplained_fraction", th.max_unexplained_fraction),
        ("gap_fraction", th.max_gap_fraction),
        ("replay_rms", th.max_replay_rms),
        ("window_failure_fraction", th.max_window_failure_fraction),
        ("saturated_window_fraction", th.max_saturated_window_fraction),
    )
    for name, limit in checks:
        value = rep[name]
        if value > limit:  # NaN compares False: no accepted samples -> no reason
            reasons.append(f"{name} {value:.4g} exceeds threshold {limit:.4g}")
    unusable = (
        rep["unexplained_fraction"] > th.unusable_unexplained_fraction
        or rep["window_failure_fraction"] > _UNUSABLE_WINDOW_FAILURE
    )
    if unusable:
        return "unusable", tuple(reasons)
    return ("review" if reasons else "qualified-software-match"), tuple(reasons)


def summarize_match_quality(
    result: DriftAnchoredMatchResult,
    thresholds: QualityThresholds = QualityThresholds(),
) -> MatchQualityReport:
    """Summarise where a match is unreliable.

    The verdict is a SOFTWARE verdict on this run only; it is not capture or
    engine qualification (epic #11421). Thresholds are provisional.

    Preconditions: result arrays share one length ``T >= 1``; thresholds are
    fractions in [0, 1] and ``max_replay_rms > 0``.
    Postcondition: ``verdict`` is one of "qualified-software-match",
    "review", "unusable" ("unusable" if unexplained fraction exceeds
    ``unusable_unexplained_fraction`` or more than half the windows failed;
    "review" if any ``max_*`` threshold is exceeded, with one reason each).

    Args:
        result: Output of ``match_kinematics``.
        thresholds: Provisional software thresholds.

    Returns:
        A :class:`MatchQualityReport`.
    """
    _validate_thresholds(thresholds)
    n = _validate_result(result)
    labels = _label_values(result)
    runs = _runs(labels)
    accepted = labels == _ACCEPTED
    frac = {lab: float(np.mean(labels == lab)) for lab in _LABELS}
    tau = np.asarray(result.tau, dtype=float)
    # Per-channel normalisation by that channel's peak |tau|.
    peak = np.maximum(np.max(np.abs(tau), axis=0), np.finfo(float).tiny)
    rel = np.asarray(result.tau_std, dtype=float) / peak[None, :]
    windows = tuple(result.windows)
    metrics = {
        "unexplained_fraction": frac[_UNEXPLAINED],
        "gap_fraction": frac[_GAP],
        "replay_rms": _rms_over(result.replay_residual, accepted),
        "window_failure_fraction": _window_fraction(windows, lambda w: not w.success),
        "saturated_window_fraction": _window_fraction(
            windows, lambda w: bool(np.any(w.saturated))
        ),
    }
    verdict, reasons = _verdict(metrics, thresholds)
    return MatchQualityReport(
        n_samples=n,
        accepted_fraction=frac[_ACCEPTED],
        outlier_fraction=frac[_OUTLIER],
        unexplained_fraction=frac[_UNEXPLAINED],
        gap_fraction=frac[_GAP],
        suspect_intervals=tuple(_interval(result, r) for r in runs),
        longest_unexplained_run=_longest(runs, _UNEXPLAINED),
        longest_gap_run=_longest(runs, _GAP),
        replay_rms=metrics["replay_rms"],
        estimate_rms=_rms_over(result.normalized_residual, accepted),
        tau_disagreement_p95=np.percentile(result.tau_disagreement, 95, axis=0),
        relative_torque_uncertainty_p95=np.percentile(rel, 95, axis=0),
        window_failure_fraction=metrics["window_failure_fraction"],
        saturated_window_fraction=metrics["saturated_window_fraction"],
        rate_limited_window_fraction=_window_fraction(
            windows, lambda w: bool(np.any(w.rate_limited))
        ),
        mean_drift_dominance=float(np.nanmean(result.drift_dominance)),
        verdict=verdict,
        reasons=reasons,
    )


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, SuspectInterval):
        return {k: _json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def report_to_dict(report: MatchQualityReport) -> dict[str, object]:
    """JSON-serialisable dict of a report (arrays to lists, NaN to None)."""
    return {f.name: _json_safe(getattr(report, f.name)) for f in fields(report)}
