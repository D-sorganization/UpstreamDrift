"""Benchmark-style tests for the recursive ZTCF-anchored kinematic matcher.

Synthetic truth is truth of the simulator, not a human measurement: these
tests establish software behaviour on the reference double pendulum. They do
not qualify any capture or engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.estimation.drift_anchored_matcher import (
    MatchOptions,
    SampleLabel,
    match_kinematics,
)
from src.shared.python.estimation.local_torque_window import WindowOptions
from src.shared.python.estimation.synthetic_swing import (
    corrupt_observations,
    reference_control_band,
)

pytestmark = pytest.mark.unit

BAND = reference_control_band()


@pytest.fixture
def provider(dime_provider):  # type: ignore[no-untyped-def]
    return dime_provider


@pytest.fixture
def truth(dime_truth):  # type: ignore[no-untyped-def]
    return dime_truth


@pytest.fixture
def corrupted_match(dime_corrupted_match):  # type: ignore[no-untyped-def]
    return dime_corrupted_match


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


class TestCleanRecovery:
    def test_low_noise_torque_profile_is_recovered(
        self, dime_options, provider, truth
    ) -> None:  # type: ignore[no-untyped-def]
        obs = corrupt_observations(
            truth, noise_std=1e-4, outlier_fraction=0.0, occlusion=None, seed=1
        )
        res = match_kinematics(
            provider, truth.t, obs.q_observed, obs.mask, BAND, dime_options(1e-4)
        )
        err = res.tau[:-1] - truth.tau[:-1]
        peak = np.max(np.abs(truth.tau), axis=0)
        assert np.all(np.sqrt(np.mean(err**2, axis=0)) < 0.05 * peak)
        assert np.all(res.labels == SampleLabel.ACCEPTED)

    def test_replay_is_one_uninterrupted_forward_simulation(
        self, dime_options, provider, truth
    ) -> None:  # type: ignore[no-untyped-def]
        from src.shared.python.estimation.local_torque_window import rollout

        obs = corrupt_observations(
            truth, noise_std=1e-4, outlier_fraction=0.0, occlusion=None, seed=1
        )
        res = match_kinematics(
            provider, truth.t, obs.q_observed, obs.mask, BAND, dime_options(1e-4)
        )
        q_ref, _ = rollout(
            provider, res.q_replay[0], res.v_replay[0], res.tau[:-1], truth.dt
        )
        np.testing.assert_allclose(res.q_replay, q_ref, atol=1e-12)
        assert _rms(res.q_replay - truth.q) < 0.02


class TestCorruptedData:
    def test_injected_outliers_are_detected(self, corrupted_match) -> None:  # type: ignore[no-untyped-def]
        obs, res = corrupted_match
        flagged = np.flatnonzero(
            np.isin(res.labels, [SampleLabel.OUTLIER, SampleLabel.UNEXPLAINED])
        )
        recall = np.isin(obs.outlier_indices, flagged).mean()
        clean = np.setdiff1d(
            np.flatnonzero(obs.mask), obs.outlier_indices, assume_unique=True
        )
        false_positive = np.isin(clean, flagged).mean()
        assert recall >= 0.9
        assert false_positive <= 0.05

    def test_occlusion_is_seen_through_by_dynamics(
        self, corrupted_match, truth
    ) -> None:  # type: ignore[no-untyped-def]
        obs, res = corrupted_match
        gap = obs.occluded_indices
        assert np.all(res.labels[gap] == SampleLabel.GAP_FILLED)
        # The dynamically consistent estimate bridges the gap far better than
        # the gross outlier scale (0.15 rad).
        assert _rms(res.q_estimate[gap] - truth.q[gap]) < 0.01

    def test_estimate_is_closer_to_truth_than_the_raw_data(
        self, corrupted_match, truth
    ) -> None:  # type: ignore[no-untyped-def]
        obs, res = corrupted_match
        seen = obs.mask
        raw = _rms(obs.q_observed[seen] - truth.q[seen])
        est = _rms(res.q_estimate[seen] - truth.q[seen])
        assert est < 0.5 * raw

    def test_torque_is_recovered_through_noise_outliers_and_gap(
        self, corrupted_match, truth
    ) -> None:  # type: ignore[no-untyped-def]
        _, res = corrupted_match
        err = res.tau[:-1] - truth.tau[:-1]
        peak = np.max(np.abs(truth.tau), axis=0)
        assert np.all(np.sqrt(np.mean(err**2, axis=0)) < 0.15 * peak)

    def test_overlay_disagreement_and_uncertainty_are_reported(
        self, corrupted_match
    ) -> None:  # type: ignore[no-untyped-def]
        _, res = corrupted_match
        assert res.tau_disagreement.shape == res.tau.shape
        assert res.tau_std.shape == res.tau.shape
        assert np.all(np.isfinite(res.tau_std))
        assert np.all(res.tau_disagreement >= 0.0)


class TestUnexplainableSegments:
    def test_persistent_marker_slip_is_flagged_not_absorbed(
        self, dime_options, provider, truth
    ) -> None:  # type: ignore[no-untyped-def]
        """A slowly slipping marker is not an isolated spike: it needs motion
        the bounded torques cannot produce, so the run is labelled suspect."""
        obs = corrupt_observations(
            truth, noise_std=0.002, outlier_fraction=0.0, occlusion=None, seed=2
        )
        q = obs.q_observed.copy()
        seg = slice(120, 140)
        q[seg, 1] += 0.4 * np.sin(np.linspace(0.0, np.pi, 20)) ** 2  # 0.4 rad slip
        res = match_kinematics(
            provider, truth.t, q, obs.mask, BAND, dime_options(0.002)
        )
        flagged = np.isin(
            res.labels[seg], [SampleLabel.OUTLIER, SampleLabel.UNEXPLAINED]
        )
        assert flagged.mean() >= 0.5
        assert np.any(res.labels[seg] == SampleLabel.UNEXPLAINED)


class TestDiagnostics:
    def test_drift_dominance_rises_with_speed(self, corrupted_match, truth) -> None:  # type: ignore[no-untyped-def]
        """Owner's note: ZTCF weight should be higher at high velocity."""
        _, res = corrupted_match
        speed = np.linalg.norm(truth.v, axis=1)
        assert np.corrcoef(speed, res.drift_dominance)[0, 1] > 0.5

    def test_ztcf_divergence_and_explained_fraction_are_per_sample(
        self, corrupted_match
    ) -> None:  # type: ignore[no-untyped-def]
        obs, res = corrupted_match
        n = obs.mask.size
        assert res.ztcf_divergence.shape == (n,)
        assert res.explained_fraction.shape == (n,)
        ok = res.labels == SampleLabel.ACCEPTED
        assert np.nanmedian(res.explained_fraction[ok]) > 0.9

    def test_every_window_has_a_record(self, corrupted_match) -> None:  # type: ignore[no-untyped-def]
        obs, res = corrupted_match
        starts = [w.start for w in res.windows]
        assert starts == sorted(starts)
        assert starts[0] == 0
        assert starts[-1] + 12 >= obs.mask.size - 1


class TestContracts:
    def test_rejects_stride_longer_than_window(self) -> None:
        with pytest.raises(ValueError, match="stride"):
            MatchOptions(
                window=WindowOptions(sigma_obs=1e-3, sigma_q0=1e-3, sigma_v0=1.0),
                window_steps=4,
                stride=5,
            )

    def test_rejects_non_uniform_time(self, dime_options, provider) -> None:  # type: ignore[no-untyped-def]
        t = np.array(
            [0.0, 0.002, 0.005, 0.006] + [0.006 + 0.002 * i for i in range(1, 30)]
        )
        q = np.zeros((t.size, 2))
        with pytest.raises(ValueError, match="uniform"):
            match_kinematics(
                provider, t, q, np.ones(t.size, bool), BAND, dime_options(1e-3)
            )

    def test_rejects_data_without_two_leading_observations(
        self, dime_options, provider
    ) -> None:  # type: ignore[no-untyped-def]
        t = 0.002 * np.arange(40)
        q = np.zeros((40, 2))
        mask = np.ones(40, bool)
        mask[1] = False
        with pytest.raises(ValueError, match="initial"):
            match_kinematics(provider, t, q, mask, BAND, dime_options(1e-3))
