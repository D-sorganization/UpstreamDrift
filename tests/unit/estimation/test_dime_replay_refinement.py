"""Tests for whole-trajectory replay refinement (DIME-09 slice, epic #11421).

The local overlay is locally consistent but its open-loop replay drifts on
noisy data. Refinement adjusts the initial state and torque knots of ONE
uninterrupted forward simulation so it matches the accepted observations,
with the overlay as prior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.drift_anchored_matcher import SampleLabel
from src.shared.python.estimation.local_torque_window import rollout
from src.shared.python.estimation.replay_refinement import (
    ReplayOptions,
    refine_match_replay,
)
from src.shared.python.estimation.synthetic_swing import reference_control_band

pytestmark = pytest.mark.unit

NOISE = 0.002  # matches the shared benchmark in conftest.py
BAND = reference_control_band()
CANDIDATES = (32, 20, 12, 8, 4)


@pytest.fixture(scope="module")
def setup(dime_provider, dime_truth, dime_corrupted_match):  # type: ignore[no-untyped-def]
    provider, truth = dime_provider, dime_truth
    obs, match = dime_corrupted_match
    refined = refine_match_replay(
        provider,
        match,
        obs.q_observed,
        BAND,
        ReplayOptions(sigma_obs=NOISE, knot_stride_candidates=CANDIDATES),
    )
    return provider, truth, obs, match, refined


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


class TestRefinement:
    def test_refined_replay_tracks_truth_much_better_than_overlay_replay(
        self, setup
    ) -> None:  # type: ignore[no-untyped-def]
        _, truth, _, match, refined = setup
        before = _rms(match.q_replay - truth.q)
        after = _rms(refined.q - truth.q)
        assert after < 0.5 * before
        assert after < 0.01

    def test_replay_is_a_single_uninterrupted_simulation(self, setup) -> None:  # type: ignore[no-untyped-def]
        provider, truth, _, _, refined = setup
        q, v = rollout(provider, refined.q[0], refined.v[0], refined.tau[:-1], truth.dt)
        np.testing.assert_allclose(refined.q, q, atol=1e-12)
        np.testing.assert_allclose(refined.v, v, atol=1e-12)

    def test_torques_respect_the_actuator_box(self, setup) -> None:  # type: ignore[no-untyped-def]
        *_, refined = setup
        assert np.all(refined.tau >= BAND.lower - 1e-9)
        assert np.all(refined.tau <= BAND.upper + 1e-9)

    def test_fit_to_accepted_observations_does_not_get_worse(self, setup) -> None:  # type: ignore[no-untyped-def]
        *_, refined = setup
        assert refined.observation_rms_after <= refined.observation_rms_before
        assert refined.success, refined.status

    def test_only_accepted_samples_are_used(self, setup) -> None:  # type: ignore[no-untyped-def]
        _, _, obs, match, refined = setup
        used = refined.used
        assert not np.any(used[obs.outlier_indices])
        assert not np.any(used[obs.occluded_indices])
        np.testing.assert_array_equal(used, match.labels == SampleLabel.ACCEPTED)

    def test_torque_recovery_improves_or_holds(self, setup) -> None:  # type: ignore[no-untyped-def]
        _, truth, _, match, refined = setup
        err_before = np.sqrt(np.mean((match.tau - truth.tau) ** 2, axis=0))
        err_after = np.sqrt(np.mean((refined.tau - truth.tau) ** 2, axis=0))
        assert np.all(err_after <= 1.1 * err_before)


class TestContracts:
    def test_options_validate(self) -> None:
        with pytest.raises(ValueError):
            ReplayOptions(sigma_obs=0.0)
        with pytest.raises(ValueError):
            ReplayOptions(sigma_obs=1e-3, knot_stride=0)

    def test_rejects_observation_shape_mismatch(self, setup) -> None:  # type: ignore[no-untyped-def]
        provider, _, obs, match, _ = setup
        with pytest.raises(ValueError, match="q_observed"):
            refine_match_replay(
                provider,
                match,
                obs.q_observed[:-1],
                BAND,
                ReplayOptions(sigma_obs=NOISE),
            )


class TestDiscrepancyKnotSelection:
    def test_selected_knots_are_noise_consistent_and_coarsest(self, setup) -> None:  # type: ignore[no-untyped-def]
        """The coarsest spacing whose residual matches the noise is chosen."""
        _, _, obs, _, refined = setup
        used = refined.used
        per_coord = np.mean(((refined.q[used] - obs.q_observed[used]) / NOISE) ** 2, 0)
        assert np.all(per_coord <= 1.0 + 3.0 * np.sqrt(2.0 / used.sum()))
        assert refined.knot_stride in CANDIDATES
        assert refined.knot_stride < max(CANDIDATES)  # coarsest under-fits a joint

    def test_fixed_fine_knots_overfit_noise(self, setup) -> None:  # type: ignore[no-untyped-def]
        """Finer knots fit closer to the noise floor: that is overfitting."""
        provider, _, obs, match, refined = setup
        finer = max(c for c in CANDIDATES if c < refined.knot_stride)
        fine = refine_match_replay(
            provider,
            match,
            obs.q_observed,
            BAND,
            ReplayOptions(sigma_obs=NOISE, knot_stride=finer),
        )
        assert fine.knot_stride == finer
        assert fine.observation_rms_after < refined.observation_rms_after
