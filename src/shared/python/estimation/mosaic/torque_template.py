"""Cross-trial torque template and motor-signature modes (MOSAIC).

Repeated trials of one subject share a phase-indexed torque pattern.  The
template ``ubar(phase)`` is both a *coupling prior* in the inner solve (see
``template_weight`` in :mod:`inner_solve`) and an interpretable output: the
player's mean motor program plus its principal modes of trial-to-trial
variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import ensure, require

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PrincipalModes:
    """Mean template, orthonormal variation modes and per-trial scores."""

    mean: FloatArray
    modes: FloatArray
    scores: FloatArray
    explained_variance_ratio: FloatArray


def phase_bins(times: FloatArray, n_bins: int) -> IntArray:
    """Map trial times to ``n_bins`` equal-width normalized-phase bins."""
    require(n_bins >= 1, "n_bins must be >= 1", n_bins)
    require(times.ndim == 1 and times.size >= 2, "need >= 2 times")
    span = float(times[-1] - times[0])
    require(span > 0.0, "times must span a positive interval")
    phase = (times - times[0]) / span
    bins = np.minimum((phase * n_bins).astype(np.int64), n_bins - 1)
    ensure(bool(bins.min() >= 0 and bins.max() < n_bins), "bins in range")
    return bins


def extract_template(
    inputs: FloatArray, trial_index: IntArray, phase_index: IntArray, n_bins: int
) -> tuple[FloatArray, FloatArray]:
    """Return ``(template (P, nu), per_trial (K, P, nu))`` by phase-bin averaging.

    Each trial is first averaged within its bins and the template is the
    *trial-balanced* mean (every trial counts once regardless of its sample
    count).  Empty (trial, bin) cells are filled with the template value so the
    per-trial array is complete.
    """
    require(inputs.ndim == 2, "inputs must be (N, nu)")
    require(trial_index.shape == (inputs.shape[0],), "trial_index per node")
    require(phase_index.shape == (inputs.shape[0],), "phase_index per node")
    n_trials = int(trial_index.max()) + 1
    n_u = inputs.shape[1]
    sums = np.zeros((n_trials, n_bins, n_u))
    counts = np.zeros((n_trials, n_bins))
    np.add.at(sums, (trial_index, phase_index), inputs)
    np.add.at(counts, (trial_index, phase_index), 1.0)
    present = counts > 0
    per_trial_means = sums / np.maximum(counts, 1.0)[..., None]
    trials_per_bin = np.maximum(present.sum(axis=0), 1)[:, None]
    template = (per_trial_means * present[..., None]).sum(axis=0) / trials_per_bin
    filled = np.where(present[..., None], per_trial_means, template[None])
    return template, filled


def principal_modes(per_trial: FloatArray, n_modes: int) -> PrincipalModes:
    """SVD of trial deviations from the mean template (flattened over phase/input)."""
    require(per_trial.ndim == 3, "per_trial must be (K, P, nu)")
    n_trials = per_trial.shape[0]
    require(
        1 <= n_modes <= max(n_trials - 1, 1), "n_modes must be in [1, K-1]", n_modes
    )
    mean = per_trial.mean(axis=0)
    deviations = (per_trial - mean[None]).reshape(n_trials, -1)
    _, singular, vt = np.linalg.svd(deviations, full_matrices=False)
    variance = singular**2
    total = float(variance.sum()) if variance.sum() > 0 else 1.0
    modes = vt[:n_modes].reshape(n_modes, *per_trial.shape[1:])
    scores = deviations @ vt[:n_modes].T
    return PrincipalModes(mean, modes, scores, variance[:n_modes] / total)
