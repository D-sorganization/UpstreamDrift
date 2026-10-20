"""Sensitivity of the forward-dynamics replay to nearby inputs (#13112).

The tracked full-body replay is deterministic: the same reference on the same
build gives a bit-identical record (fixed-step RK4, no random numbers, no
threads; ``tests/unit/motion_matching/pipeline/test_replay_sensitivity_13112.py``).
It is, however, sensitive: before impact the computed-torque tracking pulls a
perturbed run back to the reference, but after impact the ball's impulse, the
release and the single-foot follow-through let two nearby runs separate
exponentially. A sub-millimetre change of a segment's centre of mass, a 1e-16
change in a calibration (summation order of a BLAS reduction on another host)
or a receipt rounding of 1e-5 in a document is then amplified to tens of
millimetres of whole-swing marker RMS by the end of the follow-through.

That is a property of the closed-loop dynamics, not a bug to suppress. This
module measures it instead of hiding it:

* :func:`finite_time_growth_rate` fits the finite-time Lyapunov-style
  exponent ``lambda`` of ``|dx(t)| ~ |dx_0| exp(lambda t)`` over a window;
* :func:`divergence_report` compares the address-to-impact and after-impact
  windows of the separation between a nominal and a perturbed replay;
* :func:`replay_sensitivity` runs the perturbed replay and returns the
  receipt block ``dynamics.replay_sensitivity``;
* :func:`post_impact_window_rms` gives the bounded acceptance metric: the
  marker RMS over a fixed window after impact, short enough that a
  sub-millimetre input change cannot move it by more than the stated
  reproducibility tolerance;
* :func:`compare_receipts` reports which receipt values differ between two
  runs beyond a tolerance, ignoring wall-clock fields.

It reports only; no gate reads it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import math
from typing import Any

import numpy as np

#: Size of the reference perturbation of the sensitivity probe, in the units
#: of each generalized coordinate (m or rad). 1 um / 1 urad is far below the
#: capture's marker noise and below any parameter change a model edit makes,
#: and far above double-precision round-off on these magnitudes (~1e-15), so
#: the measured growth is the dynamics', not the arithmetic's.
PERTURBATION = 1e-6
#: Seed of the fixed perturbation direction (a receipt must reproduce).
PERTURBATION_SEED = 13112
#: Separations below this are round-off, not signal, and are not fitted.
SEPARATION_FLOOR_M = 1e-12
#: Stated run-to-run reproducibility tolerance of a receipt length (m):
#: 0.1 mm, two orders below the 31-34 mm address-to-impact replay RMS and the
#: smallest model-change effect the receipts are used to rank (~1 mm).
REPRODUCIBILITY_TOLERANCE_M = 1e-4
#: Fixed after-impact window of the bounded acceptance metric (s). With the
#: measured after-impact growth rates of the canonical driver and 7-iron
#: replays (``CANONICAL_RUN.md`` #13112 paragraph), a 1e-6 input change grows
#: by well under the 0.1 mm tolerance in 0.1 s, while the follow-through to
#: the end of the capture (0.5-0.6 s after impact) amplifies it to the
#: 10-80 mm swings the issue reports. 0.1 s covers the ball contact, the
#: release and the start of the follow-through: the part of the swing after
#: impact that still bears on the ball.
POST_IMPACT_WINDOW_S = 0.1
#: Receipt keys that are wall-clock or host dependent and never reproduce.
VOLATILE_KEYS = frozenset({"elapsed_s"})


def finite_time_growth_rate(
    times: Sequence[float] | np.ndarray,
    separation: Sequence[float] | np.ndarray,
    window: tuple[float, float],
    *,
    floor: float = SEPARATION_FLOOR_M,
) -> float | None:
    """Least-squares slope of ``ln |dx|`` against time over ``window`` (1/s).

    A positive value is the finite-time Lyapunov-style exponent: nearby runs
    separate as ``exp(lambda t)``. A negative value means the tracking pulls
    them together. Samples at or below ``floor`` are round-off and skipped;
    ``None`` when fewer than three samples remain.
    """
    t = np.asarray(times, dtype=float)
    sep = np.asarray(separation, dtype=float)
    if t.ndim != 1 or sep.shape != t.shape:
        raise ValueError("times and separation must be 1-D of equal length")
    if not np.isfinite(t).all() or not np.isfinite(sep).all() or (sep < 0).any():
        raise ValueError("times and separation must be finite, separation >= 0")
    lo, hi = window
    if not hi > lo:
        raise ValueError("window must have positive length")
    rows = (t >= lo) & (t <= hi) & (sep > floor)
    if rows.sum() < 3:
        return None
    slope, _ = np.polyfit(t[rows], np.log(sep[rows]), 1)
    return float(slope)


def _window_max(t: np.ndarray, sep: np.ndarray, lo: float, hi: float) -> float | None:
    rows = (t >= lo) & (t <= hi)
    return float(sep[rows].max()) if rows.any() else None


def divergence_report(
    times: Sequence[float] | np.ndarray,
    separation_m: Sequence[float] | np.ndarray,
    impact_time_s: float,
    epsilon: float,
    *,
    post_window_s: float = POST_IMPACT_WINDOW_S,
) -> dict[str, Any]:
    """Growth of the separation of two nearby replays, split at impact.

    Args:
        times: Sample times (s).
        separation_m: RMS marker distance between the nominal and the
            perturbed replay at each time (m).
        impact_time_s: Ball impact inside the time span.
        epsilon: Size of the input perturbation that produced the separation.
        post_window_s: Window of the bounded after-impact metric.

    Returns:
        Growth exponents and largest separations of the address-to-impact,
        the bounded after-impact window and the whole after-impact phase, the
        amplification ``max separation / epsilon`` of each, and the e-folding
        time after impact.
    """
    t = np.asarray(times, dtype=float)
    sep = np.asarray(separation_m, dtype=float)
    if not t[0] <= impact_time_s <= t[-1]:
        raise ValueError("impact time must lie inside the time span")
    if not epsilon > 0 or post_window_s <= 0:
        raise ValueError("epsilon and the window must be positive")
    end = float(t[-1])
    bounded_end = min(impact_time_s + post_window_s, end)
    pre = finite_time_growth_rate(t, sep, (float(t[0]), impact_time_s))
    post = finite_time_growth_rate(t, sep, (impact_time_s, end))
    pre_max = _window_max(t, sep, float(t[0]), impact_time_s)
    window_max = _window_max(t, sep, impact_time_s, bounded_end)
    post_max = _window_max(t, sep, impact_time_s, end)
    return {
        "epsilon": float(epsilon),
        "impact_time_s": float(impact_time_s),
        "post_impact_window_s": float(post_window_s),
        "growth_rate_address_to_impact_per_s": pre,
        "growth_rate_after_impact_per_s": post,
        "e_folding_time_after_impact_s": (
            1.0 / post if post is not None and post > 0 else None
        ),
        "separation_max_address_to_impact_m": pre_max,
        "separation_max_post_impact_window_m": window_max,
        "separation_max_after_impact_m": post_max,
        "amplification_address_to_impact": (
            None if pre_max is None else pre_max / epsilon
        ),
        "amplification_post_impact_window": (
            None if window_max is None else window_max / epsilon
        ),
        "amplification_after_impact": None if post_max is None else post_max / epsilon,
    }


def perturbation_direction(
    shape: tuple[int, ...], seed: int = PERTURBATION_SEED
) -> np.ndarray:
    """Fixed unit-RMS direction, constant in time, over the coordinates.

    Seeded so a receipt that records the probe reproduces. The direction is
    drawn per coordinate and applied to every frame, so the perturbed
    reference is a nearby smooth reference, not added noise.
    """
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must be (frames, coordinates)")
    direction = np.random.default_rng(seed).standard_normal(shape[1])
    direction /= math.sqrt(float(np.mean(direction**2)))
    return np.broadcast_to(direction, shape).copy()


def marker_separation(
    markers: Callable[[np.ndarray], np.ndarray], q_a: np.ndarray, q_b: np.ndarray
) -> np.ndarray:
    """Per-frame RMS distance (m) between the markers of two trajectories."""
    a, b = np.asarray(q_a, dtype=float), np.asarray(q_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("trajectories must share shape (frames, coordinates)")
    out = np.empty(a.shape[0])
    for k, (row_a, row_b) in enumerate(zip(a, b, strict=True)):
        diff = np.asarray(markers(row_a)) - np.asarray(markers(row_b))
        out[k] = math.sqrt(float(np.mean(np.einsum("ij,ij->i", diff, diff))))
    return out


def replay_sensitivity(
    run_replay: Callable[[np.ndarray], np.ndarray],
    markers: Callable[[np.ndarray], np.ndarray],
    times: Sequence[float] | np.ndarray,
    q_track: np.ndarray,
    nominal_q: np.ndarray,
    impact_time_s: float,
    *,
    epsilon: float = PERTURBATION,
    seed: int = PERTURBATION_SEED,
    post_window_s: float = POST_IMPACT_WINDOW_S,
) -> dict[str, Any]:
    """Receipt block ``dynamics.replay_sensitivity``.

    Replays ``q_track + epsilon * direction`` with ``run_replay`` (which
    returns the simulated coordinates on ``times``) and reports how the
    marker separation from ``nominal_q`` grows before and after impact.
    """
    ref = np.asarray(q_track, dtype=float)
    perturbed = ref + epsilon * perturbation_direction(ref.shape, seed)
    sim_b = np.asarray(run_replay(perturbed), dtype=float)
    separation = marker_separation(markers, nominal_q, sim_b)
    report = divergence_report(
        times, separation, impact_time_s, epsilon, post_window_s=post_window_s
    )
    report["perturbation_seed"] = int(seed)
    report["perturbation"] = "constant offset of every reference coordinate"
    return report


def post_impact_window_rms(
    times: Sequence[float] | np.ndarray,
    errors: np.ndarray,
    valid: np.ndarray,
    impact_time_s: float,
    window_s: float = POST_IMPACT_WINDOW_S,
) -> dict[str, Any]:
    """Valid-marker RMS over ``(impact, impact + window_s]`` (bounded metric).

    Unlike the whole after-impact RMS, this window ends before the replay's
    sensitivity can amplify a sub-millimetre input change past the stated
    reproducibility tolerance, so it ranks model changes.
    """
    t = np.asarray(times, dtype=float)
    err = np.asarray(errors, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if err.ndim != 2 or err.shape != mask.shape or err.shape[0] != t.shape[0]:
        raise ValueError("errors, valid and times must share shape (frames, markers)")
    if window_s <= 0:
        raise ValueError("window must be positive")
    rows = (t > impact_time_s) & (t <= impact_time_s + window_s)
    sample = err[rows][mask[rows]]
    return {
        "post_impact_window_s": float(window_s),
        "fd_rms_post_impact_window_m": (
            float(math.sqrt(float(np.mean(sample**2)))) if sample.size else None
        ),
        "post_impact_window_frames": int(rows.sum()),
    }


def _leaves(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            if key in VOLATILE_KEYS:
                continue
            yield from _leaves(value[key], f"{path}.{key}" if path else str(key))
    elif isinstance(value, list | tuple):
        yield f"{path}#len", ("len", len(value))  # exact, never toleranced
        for i, item in enumerate(value):
            yield from _leaves(item, f"{path}[{i}]")
    else:
        yield path, value


def compare_receipts(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> list[str]:
    """Paths whose values differ between two receipts.

    Numbers compare with ``|x - y| <= atol + rtol |y|`` (NaN equals NaN);
    everything else compares exactly. Wall-clock keys (``VOLATILE_KEYS``)
    are skipped. An empty list means the receipts reproduce.
    """
    left, right = dict(_leaves(a)), dict(_leaves(b))
    differ = sorted(set(left) ^ set(right))
    for key in sorted(set(left) & set(right)):
        x, y = left[key], right[key]
        numeric = isinstance(x, int | float) and isinstance(y, int | float)
        if numeric and not isinstance(x, bool) and not isinstance(y, bool):
            fx, fy = float(x), float(y)
            if math.isnan(fx) and math.isnan(fy):
                continue
            if not abs(fx - fy) <= atol + rtol * abs(fy):
                differ.append(key)
        elif x != y:
            differ.append(key)
    return differ
