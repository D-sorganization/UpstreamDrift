"""Round-trip validator for CVAE inference (issue #4003 / #034).

A coefficient sample is "valid" if running it through a forward model
(``forward_fn``) reproduces the target club trajectory to within a configured
RMSE threshold. The forward model is injected so we can use the cheap
:class:`SwingSurrogate` (Option 2) by default and fall back to the heavyweight
Simscape adapter (Option 4) when ``requires_matlab`` is set.

This module is private to :mod:`motion_matching.inverse`; the public surface
is :mod:`motion_matching.inverse.predict`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_target import ClubTarget

# A round-trip forward function maps a 1-D coefficient vector to a
# (butt, clubhead, club_quat) triple compatible with ClubTarget shapes.
# We deliberately avoid importing ClubTrajectory here -- the surrogate
# returns torch tensors and the SimscapeAdapter returns numpy. The adapter
# wrapper in predict.py converts both into a uniform ndarray triple.
RoundTripOutput = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
ForwardFn = Callable[[NDArray[np.float64]], RoundTripOutput]


@dataclass(frozen=True)
class ValidationReport:
    """Per-sample round-trip RMSE report.

    Attributes:
        rmses_m: ``(n_samples,)`` ndarray of RMSE values in metres.
        accepted: Boolean mask of which samples passed the threshold.
        threshold_m: Threshold used to compute :attr:`accepted`.
        best_index: Index of the sample with the smallest RMSE (always set,
            even if no sample passed the threshold).
    """

    rmses_m: NDArray[np.float64]
    accepted: NDArray[np.bool_]
    threshold_m: float
    best_index: int


def _check_target_shapes(target: ClubTarget) -> int:
    """Return ``n`` (number of timesteps) and validate the trajectory shapes."""
    n = int(target.time.shape[0])
    if target.butt.shape != (n, 3):
        raise ValueError(
            f"target.butt must have shape ({n}, 3); got {target.butt.shape}"
        )
    if target.clubhead.shape != (n, 3):
        raise ValueError(
            f"target.clubhead must have shape ({n}, 3); got {target.clubhead.shape}"
        )
    return n


def _round_trip_rmse(
    sample: NDArray[np.float64],
    target: ClubTarget,
    forward_fn: ForwardFn,
    n_steps: int,
) -> float:
    """Run ``forward_fn(sample)`` and return the position RMSE in metres."""
    butt_pred, clubhead_pred, _ = forward_fn(sample)
    if butt_pred.shape != (n_steps, 3) or clubhead_pred.shape != (n_steps, 3):
        raise ValueError(
            "forward_fn output shape mismatch: expected "
            f"butt/clubhead to be ({n_steps}, 3); got "
            f"{butt_pred.shape} / {clubhead_pred.shape}"
        )
    db = butt_pred - target.butt
    dc = clubhead_pred - target.clubhead
    sq = float(np.mean(np.sum(db * db, axis=1) + np.sum(dc * dc, axis=1)))
    # Average per-frame squared distance is sum-of-two RMSE^2; sqrt gives a
    # combined position RMSE that is comparable across samples.
    return float(np.sqrt(sq))


def round_trip_validate(
    samples: list[NDArray[np.float64]],
    target: ClubTarget,
    forward_fn: ForwardFn,
    *,
    rmse_threshold_m: float,
) -> ValidationReport:
    """Score ``samples`` against ``target`` via ``forward_fn``.

    Args:
        samples: List of 1-D coefficient vectors.
        target: Reference :class:`ClubTarget`.
        forward_fn: Maps a coefficient vector to ``(butt, clubhead, quat)``
            ndarrays.
        rmse_threshold_m: Acceptance threshold in metres. A sample is
            accepted iff its position RMSE is strictly less than this.

    Returns:
        :class:`ValidationReport` with per-sample RMSEs, acceptance mask,
        and the best (lowest-RMSE) index.

    Raises:
        ValueError: If ``samples`` is empty or ``rmse_threshold_m`` is
            non-positive.
    """
    if not samples:
        raise ValueError("samples must contain at least one coefficient vector")
    if not (rmse_threshold_m > 0.0):
        raise ValueError(f"rmse_threshold_m must be positive; got {rmse_threshold_m!r}")
    n_steps = _check_target_shapes(target)
    rmses = np.fromiter(
        (_round_trip_rmse(s, target, forward_fn, n_steps) for s in samples),
        dtype=np.float64,
        count=len(samples),
    )
    accepted = rmses < rmse_threshold_m
    best_index = int(np.argmin(rmses))
    return ValidationReport(
        rmses_m=rmses,
        accepted=accepted,
        threshold_m=float(rmse_threshold_m),
        best_index=best_index,
    )
