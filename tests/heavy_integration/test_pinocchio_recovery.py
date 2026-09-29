"""Heavy-integration tests for the Pinocchio recovery harness (issue #4121).

These tests run the full ``synthesize_target -> fit_swing -> assert
recovery`` loop against the real ``golfer.urdf`` + Pinocchio bindings.
They are gated on ``pytest.mark.requires_pinocchio`` and skip cleanly
when the optional engine is absent.

The optimiser (``fit_swing_pinocchio``) is tracked under PIN-FIT-DRIVER
and is expected to land in a follow-on PR. Until it does, the
harness-level test xfails with a known reason: the harness itself runs
end to end (synthesis succeeds, fit raises ``NotImplementedError``,
``RecoverySummary`` records the failure cleanly), so the gate flips
from ``xfail`` to ``pass`` the moment the optimiser appears.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytestmark = [
    pytest.mark.requires_pinocchio,
    pytest.mark.integration,
    pytest.mark.slow,
]


# --------------------------------------------------------------------------- #
# Lazy fixtures: skip the whole module when pinocchio is absent.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def mm():
    """Engine motion-matching module, skipping if pinocchio is missing."""
    try:
        import pinocchio  # noqa: F401
    except ImportError:
        pytest.skip("pinocchio not installed")
    return importlib.import_module(
        "src.engines.physics_engines.pinocchio.python.motion_matching"
    )


# --------------------------------------------------------------------------- #
# Synthesize round-trip: theta -> ClubTarget passes the canonical schema.
# --------------------------------------------------------------------------- #


def test_synthesize_returns_validated_clubtarget(mm) -> None:
    """One real synthesis trip with zero theta produces a valid target."""
    n_joints = 25  # nominal; harness will resolve from URDF anyway
    theta = np.zeros(n_joints * mm.COEFFS_PER_JOINT)
    target = mm.synthesize_target_from_coefficients(theta)
    # ClubTarget construction validates the canonical schema; reaching
    # here means every postcondition held.
    assert target.time.shape[0] >= 2
    assert target.time[0] == 0.0
    assert target.butt.shape[1] == 3
    assert target.clubhead.shape[1] == 3
    assert target.club_quat.shape[1] == 4
    assert target.source.format == "synthetic"
    # impact_idx is in [1, N]; with zero torque + neutral pose the
    # discrete clubhead derivative is essentially zero so impact_idx
    # ends up at the first samples_idx by argmax tie-break -- but in
    # *any* case it lies inside the schema window.
    assert 1 <= target.impact_idx <= target.time.shape[0]


def test_synthesize_distinct_theta_distinct_sha(mm) -> None:
    """Provenance hash discriminates different theta vectors."""
    n = 25 * mm.COEFFS_PER_JOINT
    theta_a = np.zeros(n)
    theta_b = np.zeros(n)
    theta_b[0] = 1e-3
    a = mm.synthesize_target_from_coefficients(theta_a)
    b = mm.synthesize_target_from_coefficients(theta_b)
    assert a.source.sha256 != b.source.sha256


# --------------------------------------------------------------------------- #
# Recovery harness: K=5, with the optimiser stubbed when absent.
# --------------------------------------------------------------------------- #


def _identity_fit_swing(target, _kwargs):
    """A perfect optimiser: returns ``theta_truth`` straight from provenance.

    This stub demonstrates the harness wiring end-to-end without relying
    on PIN-FIT-DRIVER. It is the *upper bound* of recovery quality --
    any real optimiser should still land within tolerance.
    """
    # The TDD oracle stores theta_truth in the provenance hash, but that
    # is one-way. We cheat by reading the tested theta out of a side
    # channel: the test below calls the harness directly and sets up
    # ``fit_swing_kwargs={'theta_truth': ...}`` for the stub to inspect.
    theta = _kwargs["theta_truth"]
    return theta, {"converged": True, "stub": True}


def test_recovery_harness_runs_with_identity_stub(mm) -> None:
    """Smoke: harness wires synthesize + fit + assert end-to-end.

    Uses an identity optimiser (perfect recovery) so we exercise every
    branch of the harness without depending on PIN-FIT-DRIVER.
    """
    n_joints = 25
    rng = np.random.default_rng(0)

    truths: list[np.ndarray] = []

    def capturing_fit(target, _kwargs):
        # Pop the next ground-truth theta from the queue.
        return truths.pop(0), {"converged": True, "stub": True}

    # Pre-seed truths so the stub can return them in order.
    bounds = mm.RecoveryHarnessOptions().bounds
    scale = 0.05  # very small so the simulator stays well-posed
    for _ in range(5):
        truths.append(mm.sample_random_theta(n_joints, rng, bounds=bounds, scale=scale))
    # The harness uses its own RNG seeded identically -> reproduces
    # exactly the same theta sequence.
    summary = mm.run_recovery_sweep(
        options=mm.RecoveryHarnessOptions(
            num_samples=5,
            n_joints=n_joints,
            seed=0,
            bounds_scale=scale,
            tolerance=1e-9,
        ),
        fit_swing=capturing_fit,
    )
    assert summary.num_samples == 5
    assert summary.num_success == 5
    assert summary.success_rate == pytest.approx(1.0)
    assert summary.median_residual_inf <= 1e-9
    # All trials carry timing info.
    for trial in summary.trials:
        assert trial.wallclock_s >= 0.0
        assert trial.error is None


@pytest.mark.xfail(
    reason="fit_swing_pinocchio not yet implemented (issue PIN-FIT-DRIVER); "
    "the harness records the NotImplementedError per trial.",
    strict=False,
)
def test_recovery_harness_default_optimiser(mm) -> None:
    """Acceptance gate: K=5 random theta within tolerance.

    Currently xfails because PIN-FIT-DRIVER has not landed. The moment
    the optimiser appears under
    ``...motion_matching.fit_swing.fit_swing_pinocchio`` this test
    flips to ``pass``: the harness already calls it and asserts
    ``residual_inf < 1e-3``.
    """
    summary = mm.run_recovery_sweep(
        options=mm.RecoveryHarnessOptions(num_samples=5, seed=0, bounds_scale=0.05),
    )
    assert summary.success_rate >= 0.8, f"Recovery summary: {summary}"
    assert summary.max_residual_inf < 1e-3
