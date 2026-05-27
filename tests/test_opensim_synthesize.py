"""Tests for OpenSim ``synthesize_target_from_coefficients`` (issue #4124).

Two layers, mirroring ``test_opensim_model_loads.py``:

1. **Pure-Python tests** — exercise the SimOut -> ClubTarget mapping,
   precondition checks, and round-trip identity using a fake
   ``simulate_with_coefficients`` patched into the engine module.  These
   run without the OpenSim Python bindings and form the contract test
   suite the optimizer code can rely on.

2. **OpenSim binding tests** (``@pytest.mark.requires_opensim``) — call
   the real engine.  Skipped when ``import opensim`` fails or when the
   simulate wrapper (issue #4120) has not yet landed.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from src.engines.physics_engines.opensim.python.motion_matching import (
    SynthOptions,
    synthesize_target_from_coefficients,
)
from src.engines.physics_engines.opensim.python.motion_matching import (
    synthesize as synth_module,
)
from src.shared.python.motion_matching.club_target import ClubTarget

# ---------------------------------------------------------------------------
# Test fixture: a tiny in-memory ``SimOut``-shaped object + a fake simulate.
# ---------------------------------------------------------------------------

_N_JOINTS = 25  # OpenSim golf humanoid coordinate count (per parity spec).
_THETA_LEN = _N_JOINTS * 7
_N_SAMPLES = 51


@dataclasses.dataclass
class _FakeSimOut:
    """Stand-in for the engine ``SimOut`` until issue #4120 lands."""

    time: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    tau: np.ndarray
    grip: np.ndarray
    grip_quat: np.ndarray
    clubhead: np.ndarray
    club_quat: np.ndarray
    solver_status: str = "success"
    duration_s: float = 0.0


def _make_fake_simout(
    n_samples: int = _N_SAMPLES,
    sim_time_s: float = 0.3,
    impact_frac: float = 0.6,
) -> _FakeSimOut:
    """Build a deterministic ``_FakeSimOut`` shaped like the OpenSim engine.

    Trajectories are smooth, finite, well-bounded, and have a clean speed
    peak near ``impact_frac * sim_time_s`` so impact detection is unambiguous.
    """
    t = np.linspace(0.0, sim_time_s, n_samples)
    n = t.size

    # Joint state: zero -- only positions matter for ClubTarget.
    n_joints = _N_JOINTS
    q = np.zeros((n, n_joints))
    qd = np.zeros((n, n_joints))
    qdd = np.zeros((n, n_joints))
    tau = np.zeros((n, n_joints))

    # Grip travels along x with a smooth bump; clubhead has a stronger
    # acceleration near impact_frac.
    impact_t = impact_frac * sim_time_s
    bump = np.exp(-((t - impact_t) ** 2) / 0.01)
    grip = np.column_stack([0.5 + 0.1 * t, 0.05 * bump, np.zeros(n)])
    clubhead = np.column_stack([0.5 + 1.5 * t**2, 0.2 * bump, 0.3 * t])

    # Club orientation: identity quaternion the whole way.
    grip_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    club_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    return _FakeSimOut(
        time=t,
        q=q,
        qd=qd,
        qdd=qdd,
        tau=tau,
        grip=grip,
        grip_quat=grip_quat,
        clubhead=clubhead,
        club_quat=club_quat,
    )


@pytest.fixture
def patch_simulate(monkeypatch: pytest.MonkeyPatch):
    """Inject a fake ``simulate_with_coefficients`` module for synthesize."""

    def _patch(
        sim_out_factory=_make_fake_simout,
        sim_options_cls: type | None = None,
    ) -> dict[str, Any]:
        captured: dict[str, Any] = {"calls": []}

        def fake_simulate(theta, options):  # noqa: ANN001
            captured["calls"].append((np.array(theta, copy=True), options))
            return sim_out_factory()

        fake_module = ModuleType(
            "src.engines.physics_engines.opensim.python.motion_matching._fake_simulate"
        )
        fake_module.simulate_with_coefficients = fake_simulate  # type: ignore[attr-defined]
        if sim_options_cls is not None:
            fake_module.SimOptions = sim_options_cls  # type: ignore[attr-defined]

        monkeypatch.setattr(
            synth_module,
            "_load_simulate_module",
            lambda: fake_module,
        )
        return captured

    return _patch


def _theta_truth(seed: int = 1234) -> np.ndarray:
    """Return a reproducible coefficient vector inside generateRandomCoefficients bounds."""
    rng = np.random.default_rng(seed)
    bounds = np.array([1000.0, 1000.0, 500.0, 500.0, 100.0, 100.0, 25.0])
    # Sample to about 30% of each bound to leave headroom for any future tighter bound.
    matrix = rng.uniform(-0.3, 0.3, size=(_N_JOINTS, 7)) * bounds
    return matrix.reshape(-1)


# ---------------------------------------------------------------------------
# Layer 1: pure-Python contract tests (no OpenSim binding required).
# ---------------------------------------------------------------------------


def test_round_trip_identity_returns_clubtarget(patch_simulate) -> None:
    """A known theta synthesizes to a fully-validated ClubTarget."""
    patch_simulate()
    theta = _theta_truth()

    target = synthesize_target_from_coefficients(theta)

    assert isinstance(target, ClubTarget)
    n = target.time.shape[0]
    assert n == _N_SAMPLES
    assert target.butt.shape == (n, 3)
    assert target.clubhead.shape == (n, 3)
    assert target.club_quat.shape == (n, 4)
    assert 1 <= target.impact_idx <= n


def test_round_trip_identity_is_deterministic(patch_simulate) -> None:
    """Same theta -> identical ClubTarget arrays on two runs."""
    patch_simulate()
    theta = _theta_truth(seed=7)

    a = synthesize_target_from_coefficients(theta)
    b = synthesize_target_from_coefficients(theta)

    np.testing.assert_array_equal(a.time, b.time)
    np.testing.assert_array_equal(a.butt, b.butt)
    np.testing.assert_array_equal(a.clubhead, b.clubhead)
    np.testing.assert_array_equal(a.club_quat, b.club_quat)
    assert a.impact_idx == b.impact_idx
    assert a.source.sha256 == b.source.sha256


def test_simout_to_clubtarget_mapping_correctness(patch_simulate) -> None:
    """SimOut.grip -> ClubTarget.butt, SimOut.clubhead -> ClubTarget.clubhead."""
    captured = patch_simulate()
    theta = _theta_truth()

    target = synthesize_target_from_coefficients(theta)
    fake = _make_fake_simout()

    np.testing.assert_allclose(target.butt, fake.grip)
    np.testing.assert_allclose(target.clubhead, fake.clubhead)
    # Quaternion normalised + sign-flipped to q[:,0] >= 0.
    np.testing.assert_allclose(np.linalg.norm(target.club_quat, axis=1), 1.0)
    assert np.all(target.club_quat[:, 0] >= 0.0)
    # The simulator was actually invoked once with the requested theta.
    assert len(captured["calls"]) == 1
    np.testing.assert_array_equal(captured["calls"][0][0].reshape(-1), theta)


def test_provenance_is_synthetic_and_hashes_theta(patch_simulate) -> None:
    """source.format=='synthetic' and sha256 is the digest of theta bytes."""
    import hashlib

    patch_simulate()
    theta = _theta_truth(seed=11)

    target = synthesize_target_from_coefficients(theta)

    assert target.source.format == "synthetic"
    assert target.source.subject_id == "synthetic"
    assert target.source.trial_id == "synthesizer_v1"
    assert target.source.filename == ""
    expected_hash = hashlib.sha256(
        np.asarray(theta, dtype=np.float64).reshape(-1).tobytes()
    ).hexdigest()
    assert target.source.sha256 == expected_hash


def test_impact_idx_aligns_with_clubhead_speed_peak(patch_simulate) -> None:
    """impact_idx points at the clubhead-velocity peak (1-indexed)."""
    patch_simulate()
    theta = _theta_truth()
    target = synthesize_target_from_coefficients(theta)

    # Reproduce expected impact: argmax of |d clubhead/dt|, 1-indexed.
    fake = _make_fake_simout()
    dt = np.diff(fake.time)
    velocity = np.diff(fake.clubhead, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    expected = int(np.argmax(speed)) + 1
    assert target.impact_idx == expected


def test_options_propagate_to_simulator(patch_simulate) -> None:
    """SynthOptions.sample_rate_hz / simulation_time_s are forwarded."""

    @dataclasses.dataclass
    class _SimOpts:
        sample_rate_hz: float = 1000.0
        simulation_time_s: float = 0.3

    captured = patch_simulate(sim_options_cls=_SimOpts)

    opts = SynthOptions(sample_rate_hz=2000.0, simulation_time_s=0.5)
    synthesize_target_from_coefficients(_theta_truth(), opts)

    assert len(captured["calls"]) == 1
    forwarded_sim_opts = captured["calls"][0][1]
    assert isinstance(forwarded_sim_opts, _SimOpts)
    assert forwarded_sim_opts.sample_rate_hz == pytest.approx(2000.0)
    assert forwarded_sim_opts.simulation_time_s == pytest.approx(0.5)


def test_options_dict_fallback_when_no_simoptions_class(patch_simulate) -> None:
    """When the engine module exposes no ``SimOptions`` class, a dict is sent."""
    captured = patch_simulate()  # no SimOptions class on the fake module

    synthesize_target_from_coefficients(_theta_truth())

    forwarded = captured["calls"][0][1]
    assert isinstance(forwarded, dict)
    assert forwarded["sample_rate_hz"] == pytest.approx(1000.0)
    assert forwarded["simulation_time_s"] == pytest.approx(0.3)


def test_failed_solver_status_raises(patch_simulate, monkeypatch) -> None:
    """``solver_status == 'failed'`` from the simulator surfaces as RuntimeError."""

    def _failed_factory():
        out = _make_fake_simout()
        return dataclasses.replace(out, solver_status="failed")

    patch_simulate(sim_out_factory=_failed_factory)

    with pytest.raises(RuntimeError, match="solver_status='failed'"):
        synthesize_target_from_coefficients(_theta_truth())


def test_simulate_module_missing_raises_importerror(monkeypatch) -> None:
    """Without the simulate wrapper, the synthesizer raises ImportError."""

    def _raise_import(*args: Any, **kwargs: Any) -> None:
        raise ImportError("simulate not yet available")

    monkeypatch.setattr(synth_module, "_load_simulate_module", _raise_import)

    with pytest.raises(ImportError, match="simulate not yet available"):
        synthesize_target_from_coefficients(_theta_truth())


# ----- Precondition / DbC checks ------------------------------------------


@pytest.mark.parametrize(
    ("bad_theta", "match"),
    [
        (np.array([1.0, 2.0]), "multiple of 7"),
        (np.array([]), "multiple of 7"),
        (np.full(_THETA_LEN, np.nan), "NaN or Inf"),
        # First coefficient out of bound: |A| > 1000.
        (
            np.concatenate(
                [
                    np.array([1500.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    np.zeros(_THETA_LEN - 7),
                ]
            ),
            "coefficient A exceeds",
        ),
        # G coefficient out of bound: |G| > 25.
        (
            np.concatenate(
                [
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0]),
                    np.zeros(_THETA_LEN - 7),
                ]
            ),
            "coefficient G exceeds",
        ),
    ],
)
def test_invalid_theta_raises(bad_theta, match, patch_simulate) -> None:
    patch_simulate()
    with pytest.raises(ValueError, match=match):
        synthesize_target_from_coefficients(bad_theta)


def test_invalid_options_raise(patch_simulate) -> None:
    patch_simulate()
    with pytest.raises(ValueError, match="sample_rate_hz"):
        synthesize_target_from_coefficients(
            _theta_truth(), SynthOptions(sample_rate_hz=0.0)
        )
    with pytest.raises(ValueError, match="simulation_time_s"):
        synthesize_target_from_coefficients(
            _theta_truth(), SynthOptions(simulation_time_s=2.0)
        )


def test_quaternion_normalisation(patch_simulate) -> None:
    """A non-unit quaternion in SimOut is normalised; q[:,0] >= 0."""

    def _bad_quat_factory():
        out = _make_fake_simout()
        bad = np.tile(np.array([-2.0, 0.0, 0.0, 0.0]), (out.time.size, 1))
        return dataclasses.replace(out, club_quat=bad)

    patch_simulate(sim_out_factory=_bad_quat_factory)
    target = synthesize_target_from_coefficients(_theta_truth())

    np.testing.assert_allclose(np.linalg.norm(target.club_quat, axis=1), 1.0)
    assert np.all(target.club_quat[:, 0] >= 0.0)


def test_position_noise_is_deterministic(patch_simulate) -> None:
    """``add_noise`` perturbs positions but stays reproducible across runs."""
    patch_simulate()
    theta = _theta_truth()
    opts_noise = SynthOptions(add_noise=True, noise_sigma_m=1.0e-3)

    a = synthesize_target_from_coefficients(theta, opts_noise)
    b = synthesize_target_from_coefficients(theta, opts_noise)
    np.testing.assert_array_equal(a.butt, b.butt)
    np.testing.assert_array_equal(a.clubhead, b.clubhead)

    clean = synthesize_target_from_coefficients(theta, SynthOptions(add_noise=False))
    # Noise is non-zero almost surely; assert positions actually changed.
    assert not np.allclose(a.butt, clean.butt)


# ---------------------------------------------------------------------------
# Layer 2: live OpenSim test (skipped without bindings or simulate wrapper).
# ---------------------------------------------------------------------------


def _real_simulate_module_available() -> bool:
    """Return True iff the issue-#4120 simulate wrapper has landed."""
    candidates = (
        "src.engines.physics_engines.opensim.python.motion_matching.simulate",
        "src.engines.physics_engines.opensim.python.opensim_golf."
        "simulate_with_coefficients",
    )
    return any(importlib.util.find_spec(dotted) is not None for dotted in candidates)


@pytest.mark.requires_opensim
@pytest.mark.skipif(
    not _real_simulate_module_available(),
    reason=(
        "OpenSim simulate_with_coefficients not yet available (pending issue #4120)."
    ),
)
def test_synthesize_with_real_opensim_simulator() -> None:
    """End-to-end smoke: real engine produces a finite, schema-valid ClubTarget."""
    # Use a small, gentle coefficient vector so the model stays inside the
    # numerically stable region irrespective of the model build.
    theta = np.zeros(_THETA_LEN)
    target = synthesize_target_from_coefficients(theta)

    assert isinstance(target, ClubTarget)
    assert np.all(np.isfinite(target.butt))
    assert np.all(np.isfinite(target.clubhead))
    assert target.source.format == "synthetic"
