"""Tests for the cross-trial torque template / motor-signature analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.torque_template import (
    extract_template,
    phase_bins,
    principal_modes,
)

pytestmark = pytest.mark.unit


def test_phase_bins_cover_trial_uniformly() -> None:
    times = np.linspace(0.0, 2.0, 41)
    bins = phase_bins(times, n_bins=8)
    assert bins.min() == 0 and bins.max() == 7
    counts = np.bincount(bins, minlength=8)
    assert counts.max() - counts.min() <= 1
    with pytest.raises(ContractViolationError):
        phase_bins(times, n_bins=0)


def test_template_and_modes_recover_planted_structure() -> None:
    rng = np.random.default_rng(3)
    n_trials, n_bins, n_u = 6, 20, 2
    phase = np.linspace(0, 1, n_bins, endpoint=False)
    mean = np.stack(
        [np.sin(2 * np.pi * phase), 0.5 * np.cos(2 * np.pi * phase)], axis=1
    )
    mode = np.stack([np.cos(4 * np.pi * phase), np.zeros(n_bins)], axis=1)
    scores = rng.normal(size=n_trials)
    per_trial = mean[None] + scores[:, None, None] * mode[None]
    # unequal trial lengths: resample each trial onto its own time axis
    inputs, trial_index, phase_index = [], [], []
    for k in range(n_trials):
        n_t = 40 + 5 * k
        bins = phase_bins(np.linspace(0, 1, n_t), n_bins)
        inputs.append(per_trial[k][bins])
        trial_index.append(np.full(n_t, k))
        phase_index.append(bins)
    inputs_all = np.concatenate(inputs)
    template, per_trial_hat = extract_template(
        inputs_all, np.concatenate(trial_index), np.concatenate(phase_index), n_bins
    )
    np.testing.assert_allclose(template, mean + scores.mean() * mode, atol=1e-12)
    assert per_trial_hat.shape == (n_trials, n_bins, n_u)
    modes = principal_modes(per_trial_hat, n_modes=2)
    assert modes.explained_variance_ratio[0] > 0.999
    aligned = np.sign(modes.modes[0].ravel() @ mode.ravel()) * modes.modes[0]
    np.testing.assert_allclose(aligned, mode / np.linalg.norm(mode), atol=1e-9)
