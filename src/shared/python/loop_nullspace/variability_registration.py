"""Phase-registered and physical-time residuals side by side (NSX-97).

Event-anchored phase registration (MOSAIC-10) removes timing variability;
that is useful for shape but would hide timing variance that is itself a
physical result.  This report keeps **both** residual sets: residuals about
the phase-registered template, and residuals about the mean in physical time
measured from the impact event, plus the per-trial inter-event durations.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.shared.python.estimation.mosaic.swing_registration import (
    SWING_EVENT_ORDER,
    MultiSwingTrial,
    SwingEvent,
    align_trials_to_phase,
    evaluate_registration_variance,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RegistrationResiduals:
    """Residual arrays (trials x bins x ...) for both time bases."""

    phase_residuals: Array
    physical_residuals: Array
    physical_grid_s: Array
    phase_rms: float
    physical_rms: float
    event_durations: Array
    event_duration_sd: Array
    variance_raw: float
    variance_registered: float


def registration_residual_report(
    trials: Sequence[MultiSwingTrial], trajectory_key: str = "inputs", n_bins: int = 50
) -> RegistrationResiduals:
    """Residuals in canonical phase and in physical time about impact.

    Raises:
        ValueError: If fewer than two trials, ``n_bins < 2``, or the trials
            share no common physical window around impact.
    """
    if len(trials) < 2 or n_bins < 2:
        raise ValueError("need >= 2 trials and n_bins >= 2")
    aligned = align_trials_to_phase(trials, trajectory_key, n_bins)
    phase_res = aligned.per_trial - aligned.template

    rel = [t.times - t.events[SwingEvent.IMPACT] for t in trials]
    lo, hi = max(float(r[0]) for r in rel), min(float(r[-1]) for r in rel)
    if not hi > lo:
        raise ValueError("trials share no common physical-time window about impact")
    grid = np.asarray(np.linspace(lo, hi, n_bins), dtype=np.float64)
    resampled = []
    for r, trial in zip(rel, trials, strict=True):
        flat = trial.trajectories[trajectory_key].reshape(r.size, -1)
        cols = [np.interp(grid, r, flat[:, c]) for c in range(flat.shape[1])]
        resampled.append(
            np.stack(cols, axis=1).reshape(
                n_bins, *trial.trajectories[trajectory_key].shape[1:]
            )
        )
    stacked = np.stack(resampled)
    phys_res = stacked - stacked.mean(axis=0)
    durations = np.array(
        [
            [t.events[b] - t.events[a] for a, b in pairwise(SWING_EVENT_ORDER)]
            for t in trials
        ]
    )
    raw, reg = evaluate_registration_variance(trials, trajectory_key, n_bins)
    return RegistrationResiduals(
        phase_residuals=phase_res,
        physical_residuals=phys_res,
        physical_grid_s=grid,
        phase_rms=float(np.sqrt(np.mean(phase_res**2))),
        physical_rms=float(np.sqrt(np.mean(phys_res**2))),
        event_durations=durations,
        event_duration_sd=durations.std(axis=0, ddof=1),
        variance_raw=raw,
        variance_registered=reg,
    )
