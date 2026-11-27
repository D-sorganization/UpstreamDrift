"""6-component root-wrench residual audit (#15808 WP-0a).

The canonical-v2 swing has an unactuated root, so the generalized force that
inverse dynamics asks of the root (``fx, fy, fz, mx, my, mz``) is the
*unexplained* wrench: whatever the contacts, grip weld and actuators failed to
supply. WP-1 admits a baseline only if it is below 2% body weight (forces) and
2% body weight x height (moments) outside the impact window.

This module is pure and backend-free. The caller supplies the residual
trajectory and optional candidate-cause signals (foot slip/penetration, grip-weld
gap, inertia and differentiated-kinematics proxies); it returns per-component
gate results, the Pearson correlation of each component against each cause, a
dominant cause per component, and a machine-readable refit recommendation.
Correlation is diagnostic, not causal proof: it ranks which refit to try first.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.optimization._finite_array import finite_real_array

SCHEMA = "root_wrench_audit/v1"
COMPONENTS: tuple[str, ...] = ("fx", "fy", "fz", "mx", "my", "mz")
FORCE_GATE_BW = 0.02
MOMENT_GATE_BW_HEIGHT = 0.02
ATTRIBUTION_MIN_ABS_R = 0.5
UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class ComponentAudit:
    """Gate result for one root-wrench component (normalized units)."""

    name: str
    rms_normalized: float
    max_normalized: float
    impact_peak_normalized: float
    n_gated_frames: int
    passes: bool
    cause_correlation: Mapping[str, float]
    dominant_cause: str


@dataclass(frozen=True)
class RootWrenchAudit:
    """Machine-readable diagnosis for all six components."""

    components: tuple[ComponentAudit, ...]
    admitted: bool
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        """JSON-ready form; tolerances are recorded with the verdict."""
        return {
            "schema": SCHEMA,
            "gates": {
                "force_bw": FORCE_GATE_BW,
                "moment_bw_height": MOMENT_GATE_BW_HEIGHT,
            },
            "admitted": self.admitted,
            "recommendation": self.recommendation,
            "components": [
                {
                    "name": c.name,
                    "rms_normalized": c.rms_normalized,
                    "max_normalized": c.max_normalized,
                    "impact_peak_normalized": c.impact_peak_normalized,
                    "n_gated_frames": c.n_gated_frames,
                    "passes": c.passes,
                    "cause_correlation": dict(c.cause_correlation),
                    "dominant_cause": c.dominant_cause,
                }
                for c in self.components
            ],
        }


def _finite(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Return a finite real array, or raise ValueError naming ``name``."""
    return finite_real_array(value, name)


def _positive(value: float, name: str) -> float:
    """Return ``value`` when it is finite and positive."""
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


def _abs_pearson(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """|Pearson r|; 0.0 when either series is constant (undefined -> no link)."""
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    if denom <= 1e-300:
        return 0.0
    return min(1.0, abs(float(np.dot(xc, yc))) / denom)


def audit_root_wrench(
    *,
    times_s: ArrayLike,
    root_wrench: ArrayLike,
    body_weight_n: float,
    height_m: float,
    impact_time_s: float,
    impact_window_s: float,
    causes: Mapping[str, ArrayLike],
) -> RootWrenchAudit:
    """Audit the unexplained root wrench against the WP-1 admission gates.

    Preconditions: ``times_s`` strictly increasing 1D (N >= 2); ``root_wrench``
    finite (N, 6) ordered ``COMPONENTS`` in N and N*m; positive finite body
    weight (N) and height (m); finite ``impact_time_s``; ``impact_window_s`` >= 0;
    every cause a finite length-N series. Frames within ``impact_window_s`` of
    impact are reported (``impact_peak_normalized``) but excluded from gating
    and correlation, matching "outside impact" in the contract.

    Postconditions: a component passes iff its max |residual| over gated frames
    is <= its gate. ``admitted`` iff all six pass. ``recommendation`` is
    ``"none"`` when admitted, else ``"refit:<cause>"`` for the dominant cause of
    the worst-failing component (``unattributed`` if no cause reaches
    |r| >= 0.5).
    """
    times = _finite(times_s, "times_s")
    wrench = _finite(root_wrench, "root_wrench")
    if times.ndim != 1 or times.size < 2 or not np.all(np.diff(times) > 0.0):
        raise ValueError("times_s must be 1D, length >= 2 and strictly increasing")
    if wrench.shape != (times.size, 6):
        raise ValueError(f"root_wrench must have shape (N, 6), got {wrench.shape}")
    weight = _positive(body_weight_n, "body_weight_n")
    height = _positive(height_m, "height_m")
    if not np.isfinite(impact_time_s):
        raise ValueError("impact_time_s must be finite")
    if not (np.isfinite(impact_window_s) and impact_window_s >= 0.0):
        raise ValueError("impact_window_s must be finite and nonnegative")
    series: dict[str, NDArray[np.float64]] = {}
    for key, values in causes.items():
        arr = _finite(values, f"causes[{key!r}]")
        if arr.shape != times.shape:
            raise ValueError(f"causes[{key!r}] must have shape ({times.size},)")
        series[key] = arr

    in_impact = np.abs(times - impact_time_s) <= impact_window_s
    gated = ~in_impact
    if not gated.any():
        raise ValueError("no frames remain outside the impact window")

    scale = np.array([weight] * 3 + [weight * height] * 3)
    gates = np.array([FORCE_GATE_BW] * 3 + [MOMENT_GATE_BW_HEIGHT] * 3)
    norm = np.abs(wrench) / scale

    components = []
    worst, worst_ratio = 0, -1.0
    for i, name in enumerate(COMPONENTS):
        g = norm[gated, i]
        corr = {k: _abs_pearson(wrench[gated, i], s[gated]) for k, s in series.items()}
        top = max(corr, key=corr.__getitem__) if corr else None
        dominant = top if top and corr[top] >= ATTRIBUTION_MIN_ABS_R else UNATTRIBUTED
        ratio = float(g.max() / gates[i])
        if ratio > worst_ratio:
            worst, worst_ratio = i, ratio
        components.append(
            ComponentAudit(
                name=name,
                rms_normalized=float(np.sqrt(np.mean(g**2))),
                max_normalized=float(g.max()),
                impact_peak_normalized=float(norm[in_impact, i].max())
                if in_impact.any()
                else 0.0,
                n_gated_frames=int(g.size),
                passes=bool(g.max() <= gates[i]),
                cause_correlation=corr,
                dominant_cause=dominant,
            )
        )
    admitted = all(c.passes for c in components)
    recommendation = "none" if admitted else f"refit:{components[worst].dominant_cause}"
    return RootWrenchAudit(tuple(components), admitted, recommendation)
