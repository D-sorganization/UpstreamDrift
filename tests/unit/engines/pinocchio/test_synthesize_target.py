"""Unit tests for the Pinocchio synthesize oracle (issue #4121).

These exercise the ``SimOut -> ClubTarget`` adapter and the surrounding
contract -- with ``simulate_with_coefficients`` mocked -- so they run
without requiring the optional ``pinocchio`` extra.

The heavy round-trip path (real URDF + real ABA + recovery loop) lives
in ``tests/heavy_integration/test_pinocchio_recovery.py`` under the
``requires_pinocchio`` marker.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import numpy as np
import pytest

simulate_mod = importlib.import_module(
    "src.engines.physics_engines.pinocchio.python.motion_matching.simulate"
)
synthesize_mod = importlib.import_module(
    "src.engines.physics_engines.pinocchio.python.motion_matching.synthesize"
)
SimOut = simulate_mod.SimOut
COEFFS_PER_JOINT = simulate_mod.COEFFS_PER_JOINT
synthesize_target_from_coefficients = synthesize_mod.synthesize_target_from_coefficients
SynthesizeOptions = synthesize_mod.SynthesizeOptions


pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers: build a synthetic SimOut without touching pinocchio.
# --------------------------------------------------------------------------- #


def _identity_quat_track(n: int) -> np.ndarray:
    """``(n, 3, 3)`` stack of identity rotation matrices."""
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()


def _smooth_clubhead_track(
    n: int, peak_idx: int = 50, dt: float = 1.0e-3
) -> np.ndarray:
    """``(n, 3)`` clubhead positions whose linear-speed argmax is ``peak_idx``.

    A simple Gaussian-bump profile in the world-X channel: position is
    cumulative trapezoidal of a velocity bump centred on ``peak_idx``,
    so ``np.argmax(np.linalg.norm(diff, axis=1))`` lands on
    ``peak_idx - 1`` and our 1-based ``impact_idx`` is ``peak_idx``.
    """
    t = np.arange(n)
    sigma = max(2.0, n / 40.0)
    velocity = np.exp(-0.5 * ((t - peak_idx) / sigma) ** 2)
    pos_x = np.cumsum(velocity) * dt + 1.0  # offset so |r| > 0 but < 5
    track = np.zeros((n, 3))
    track[:, 0] = pos_x
    track[:, 2] = 0.5  # clubhead a bit above the floor
    return track


def _make_sim_out(
    n_samples: int = 101, n_joints: int = 25, dt: float = 1.0e-3
) -> SimOut:
    """Construct a minimal but schema-valid :class:`SimOut`."""
    t = np.arange(n_samples, dtype=np.float64) * dt
    nq = n_joints
    q = np.zeros((n_samples, nq), dtype=np.float64)
    qd = np.zeros((n_samples, n_joints), dtype=np.float64)
    tau = np.zeros((n_samples, n_joints), dtype=np.float64)
    grip_pos = np.tile(np.array([0.0, 0.0, 1.0]), (n_samples, 1))
    grip_rot = _identity_quat_track(n_samples)
    clubhead_pos = _smooth_clubhead_track(n_samples)
    clubhead_rot = _identity_quat_track(n_samples)
    return SimOut(
        t=t,
        q=q,
        qd=qd,
        tau=tau,
        grip_position=grip_pos,
        grip_rotation=grip_rot,
        clubhead_position=clubhead_pos,
        clubhead_rotation=clubhead_rot,
        kinetic_energy=np.zeros(n_samples),
        potential_energy=np.zeros(n_samples),
        meta={"n_joints": n_joints, "model_nq": nq, "model_nv": n_joints},
    )


# --------------------------------------------------------------------------- #
# SimOut -> ClubTarget mapping (pure adapter, no simulator).
# --------------------------------------------------------------------------- #


class TestSimOutToClubTarget:
    """Direct tests of ``_sim_out_to_club_target`` -- the schema mapping."""

    def test_maps_time_grip_clubhead_arrays(self) -> None:
        sim = _make_sim_out()
        theta = np.zeros(25 * COEFFS_PER_JOINT)
        target = synthesize_mod._sim_out_to_club_target(
            sim, theta=theta, subject_id="sub", trial_id="trial"
        )
        np.testing.assert_array_equal(target.time, sim.t)
        np.testing.assert_array_equal(target.butt, sim.grip_position)
        np.testing.assert_array_equal(target.clubhead, sim.clubhead_position)

    def test_quaternion_unit_norm_and_w_nonneg(self) -> None:
        sim = _make_sim_out()
        theta = np.zeros(25 * COEFFS_PER_JOINT)
        target = synthesize_mod._sim_out_to_club_target(
            sim, theta=theta, subject_id="sub", trial_id="trial"
        )
        norms = np.linalg.norm(target.club_quat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-9)
        # Identity rotation -> [1, 0, 0, 0]; w must be >= 0 by canonical sign.
        assert np.all(target.club_quat[:, 0] >= 0.0)
        np.testing.assert_allclose(target.club_quat[0], [1.0, 0.0, 0.0, 0.0])

    def test_impact_idx_matches_speed_argmax(self) -> None:
        peak = 47
        sim = _make_sim_out()
        # Re-synthesize with a known peak.
        clubhead = _smooth_clubhead_track(sim.t.shape[0], peak_idx=peak)
        sim2 = SimOut(
            t=sim.t,
            q=sim.q,
            qd=sim.qd,
            tau=sim.tau,
            grip_position=sim.grip_position,
            grip_rotation=sim.grip_rotation,
            clubhead_position=clubhead,
            clubhead_rotation=sim.clubhead_rotation,
            kinetic_energy=sim.kinetic_energy,
            potential_energy=sim.potential_energy,
            meta=sim.meta,
        )
        theta = np.zeros(25 * COEFFS_PER_JOINT)
        target = synthesize_mod._sim_out_to_club_target(
            sim2, theta=theta, subject_id="s", trial_id="t"
        )
        assert target.impact_idx == peak

    def test_provenance_format_and_sha_reproducible(self) -> None:
        sim = _make_sim_out()
        theta = np.linspace(-1.0, 1.0, 25 * COEFFS_PER_JOINT)
        t1 = synthesize_mod._sim_out_to_club_target(
            sim, theta=theta, subject_id="sub", trial_id="trial"
        )
        t2 = synthesize_mod._sim_out_to_club_target(
            sim, theta=theta.copy(), subject_id="sub", trial_id="trial"
        )
        assert t1.source.format == "synthetic"
        assert t1.source.filename == "synthetic"
        assert t1.source.subject_id == "sub"
        assert t1.source.trial_id == "trial"
        assert len(t1.source.sha256) == 64
        assert t1.source.sha256 == t2.source.sha256

    def test_distinct_theta_distinct_sha(self) -> None:
        sim = _make_sim_out()
        theta_a = np.zeros(25 * COEFFS_PER_JOINT)
        theta_b = np.zeros(25 * COEFFS_PER_JOINT)
        theta_b[0] = 1.0
        ta = synthesize_mod._sim_out_to_club_target(
            sim, theta=theta_a, subject_id="s", trial_id="t"
        )
        tb = synthesize_mod._sim_out_to_club_target(
            sim, theta=theta_b, subject_id="s", trial_id="t"
        )
        assert ta.source.sha256 != tb.source.sha256


# --------------------------------------------------------------------------- #
# Public entrypoint, with the simulator mocked.
# --------------------------------------------------------------------------- #


class TestSynthesizeWithMockedSimulator:
    """Ensures ``synthesize_target_from_coefficients`` only uses
    documented :class:`SimOut` fields and produces a validated
    :class:`ClubTarget` regardless of pinocchio availability."""

    def _patched(self):
        sim = _make_sim_out()

        def fake_simulate(theta, options=None, initial_pose=None):
            # Mirror the validation that ``simulate_with_coefficients``
            # performs so we still cover the contract.
            theta_arr = np.asarray(theta, dtype=np.float64)
            assert theta_arr.shape == (25 * COEFFS_PER_JOINT,)
            return sim

        return patch.object(synthesize_mod, "simulate_with_coefficients", fake_simulate)

    def test_returns_validated_clubtarget(self) -> None:
        theta = np.zeros(25 * COEFFS_PER_JOINT)
        with self._patched():
            target = synthesize_target_from_coefficients(theta)
        # ``ClubTarget.__post_init__`` ran the schema check; reaching
        # here means every postcondition held.
        assert target.time.shape[0] == 101
        assert target.butt.shape == (101, 3)
        assert target.clubhead.shape == (101, 3)
        assert target.club_quat.shape == (101, 4)
        assert 1 <= target.impact_idx <= 101
        assert target.source.format == "synthetic"

    def test_round_trip_determinism(self) -> None:
        theta = np.linspace(-0.1, 0.1, 25 * COEFFS_PER_JOINT)
        with self._patched():
            t1 = synthesize_target_from_coefficients(theta)
            t2 = synthesize_target_from_coefficients(theta.copy())
        np.testing.assert_array_equal(t1.time, t2.time)
        np.testing.assert_array_equal(t1.butt, t2.butt)
        np.testing.assert_array_equal(t1.clubhead, t2.clubhead)
        np.testing.assert_array_equal(t1.club_quat, t2.club_quat)
        assert t1.source.sha256 == t2.source.sha256
        assert t1.impact_idx == t2.impact_idx

    def test_default_trial_id_is_theta_hash_prefix(self) -> None:
        theta = np.zeros(25 * COEFFS_PER_JOINT)
        with self._patched():
            target = synthesize_target_from_coefficients(theta)
        assert target.source.trial_id.startswith("theta_")
        assert target.source.trial_id[len("theta_") :] == target.source.sha256[:8]

    def test_custom_subject_and_trial_id(self) -> None:
        theta = np.zeros(25 * COEFFS_PER_JOINT)
        opts = SynthesizeOptions(subject_id="TW", trial_id="trial_42")
        with self._patched():
            target = synthesize_target_from_coefficients(theta, options=opts)
        assert target.source.subject_id == "TW"
        assert target.source.trial_id == "trial_42"

    @pytest.mark.parametrize(
        ("bad_theta", "match"),
        [
            (np.array([], dtype=np.float64), "non-empty"),
            (np.ones(25 * COEFFS_PER_JOINT - 1), "multiple of"),
            (np.full(25 * COEFFS_PER_JOINT, np.nan), "non-finite"),
        ],
    )
    def test_rejects_malformed_theta(self, bad_theta: np.ndarray, match: str) -> None:
        with self._patched(), pytest.raises(ValueError, match=match):
            synthesize_target_from_coefficients(bad_theta)


# --------------------------------------------------------------------------- #
# Pure helpers: impact detection and theta hashing.
# --------------------------------------------------------------------------- #


class TestImpactDetection:
    def test_one_based_argmax_of_speed(self) -> None:
        track = _smooth_clubhead_track(101, peak_idx=42)
        idx = synthesize_mod._impact_idx_from_clubhead(track)
        assert idx == 42  # 1-based

    def test_integer_tracks_use_float_accumulation(self) -> None:
        track = np.array(
            [
                [0, 0, 0],
                [50_000, 0, 0],
                [50_010, 0, 0],
            ],
            dtype=np.int32,
        )

        idx = synthesize_mod._impact_idx_from_clubhead(track)

        assert idx == 1

    def test_rejects_short_tracks(self) -> None:
        with pytest.raises(ValueError, match=">= 2 samples"):
            synthesize_mod._impact_idx_from_clubhead(np.zeros((1, 3)))

    def test_rejects_non_xyz_tracks(self) -> None:
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            synthesize_mod._impact_idx_from_clubhead(np.zeros((10, 4)))


class TestThetaHash:
    def test_sha_is_64_hex_chars(self) -> None:
        sha = synthesize_mod._theta_sha256(np.arange(7, dtype=np.float64))
        assert len(sha) == 64
        int(sha, 16)  # parses as hex

    def test_sha_independent_of_array_strides(self) -> None:
        base = np.linspace(-1.0, 1.0, 100, dtype=np.float64)
        sliced = base[::2].copy()
        non_contig = np.ascontiguousarray(base)[::2]
        assert synthesize_mod._theta_sha256(sliced) == synthesize_mod._theta_sha256(
            non_contig
        )


# --------------------------------------------------------------------------- #
# SynthesizeOptions / SimOptions wiring.
# --------------------------------------------------------------------------- #


class TestSimOptionsFromAlign:
    def test_dt_is_inverse_sample_rate(self) -> None:
        from src.shared.python.motion_matching.club_target import AlignOptions

        sim_opts = synthesize_mod._sim_options_from_align(
            AlignOptions(sample_rate_hz=500.0, simulation_time_s=0.2)
        )
        assert sim_opts.dt == pytest.approx(2e-3)
        assert sim_opts.t_final == pytest.approx(0.2)
        assert sim_opts.integrator == "rk4"

    def test_rejects_non_positive_rate(self) -> None:
        from src.shared.python.motion_matching.club_target import AlignOptions

        with pytest.raises(ValueError, match="sample_rate_hz"):
            synthesize_mod._sim_options_from_align(
                AlignOptions(sample_rate_hz=0.0, simulation_time_s=0.2)
            )


# --------------------------------------------------------------------------- #
# Sanity: importing synthesize must not pull in pinocchio.
# --------------------------------------------------------------------------- #


def test_import_does_not_load_pinocchio() -> None:
    # Cheap: if any test above accidentally imported pinocchio, we'd see
    # it in sys.modules. The module should defer the import to call time.
    assert "pinocchio" not in sys.modules or sys.modules["pinocchio"] is not None
    # The check that matters: synthesize.py itself does not import pinocchio
    # at module load. Re-import in a clean state to verify.
    import importlib as _il

    _il.reload(synthesize_mod)
    # If pinocchio is not on the system, it must still not be required.
    # Conversely, if it *is* installed, this assertion still holds: we
    # only assert that the synthesize *module* did not import it.
    assert "pinocchio" not in synthesize_mod.__dict__
