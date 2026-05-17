"""Cross-language parity tests for the shared motion-matching package.

Issue #4095. The MATLAB ``compute_cost.m`` is the reference implementation;
the Python mirror in ``shared.python.motion_matching.final_cost`` must agree
numerically to within 1e-6 RMSE on a fixed fixture.

We do not call MATLAB at test time (it isn't reliably present on CI). The
parity contract is established by computing the cost analytically from the
formulas in ``COST_FUNCTION_SPEC.md`` for fixtures whose answers are
derivable to machine precision -- this is exactly the strategy used by the
existing ``test_cost.py`` and is the contract documented in MATLAB issue
parity comments. The numerical-equivalence assertion uses RMSE against the
analytic answer, with the spec's 1e-6 tolerance.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from src.shared.python.motion_matching import (
    ClubTarget,
    CostOptions,
    SimOut,
    SimOutput,
    SourceProvenance,
    compute_cost,
    compute_total_work,
    must_be_unit_quaternion_rows,
)

# Spec'd numeric tolerance for cross-language parity (issue #4095).
PARITY_RMSE_TOL = 1.0e-6


def _provenance() -> SourceProvenance:
    return SourceProvenance(
        filename="parity.bin",
        format="synthetic",
        subject_id="PARITY",
        trial_id="0",
        sha256=hashlib.sha256(b"parity-fixture-v1").hexdigest(),
    )


def _fixture_target(n: int = 41) -> ClubTarget:
    """Deterministic fixture target used by both Python and MATLAB sides."""
    time = np.linspace(0.0, 0.4, n)
    butt = np.column_stack(
        [
            0.5 * np.cos(2 * np.pi * time),
            0.5 * np.sin(2 * np.pi * time),
            0.05 * time,
        ]
    )
    clubhead = butt + np.array([0.0, 0.0, 1.1])
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return ClubTarget(
        time=time,
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=n // 2,
        source=_provenance(),
    )


def _matching_sim(target: ClubTarget) -> SimOutput:
    """SimOutput that exactly reproduces the target -- residual = 0."""
    n = target.time.shape[0]
    return SimOutput(
        butt=target.butt.copy(),
        clubhead=target.clubhead.copy(),
        club_quat=target.club_quat.copy(),
        time=target.time.copy(),
        tau=np.zeros((n, 3)),
        omega=np.zeros((n, 3)),
    )


def _shifted_sim(target: ClubTarget, shift: np.ndarray) -> SimOutput:
    """Constant-offset sim -- per-frame position error == ``shift``."""
    n = target.time.shape[0]
    return SimOutput(
        butt=target.butt + shift,
        clubhead=target.clubhead + shift,
        club_quat=target.club_quat.copy(),
        time=target.time.copy(),
        tau=np.zeros((n, 3)),
        omega=np.zeros((n, 3)),
    )


def test_sim_out_alias_is_simoutput() -> None:
    """``SimOut`` is the canonical engine-agnostic name; backed by SimOutput."""
    assert SimOut is SimOutput


def test_validators_reject_nonunit_quaternion() -> None:
    """Validator ``must_be_unit_quaternion_rows`` matches the MATLAB validator."""
    with pytest.raises(ValueError):
        must_be_unit_quaternion_rows(np.array([[1.0, 0.0, 0.0, 1.0]]))
    must_be_unit_quaternion_rows(np.array([[1.0, 0.0, 0.0, 0.0]]))


def test_compute_cost_zero_when_sim_matches_target() -> None:
    """Analytic ground truth: identical sim => J == 0 exactly (to 1e-12)."""
    target = _fixture_target()
    j, terms = compute_cost(
        np.zeros(7),
        target,
        sim_fn=lambda _theta: _matching_sim(target),
        opts=CostOptions(lambda_=0.0),
    )
    assert j == pytest.approx(0.0, abs=1e-12)
    assert terms.position == pytest.approx(0.0, abs=1e-12)
    assert terms.orientation == pytest.approx(0.0, abs=1e-12)
    assert terms.impact_anchor == pytest.approx(0.0, abs=1e-12)
    assert terms.regularizer == pytest.approx(0.0, abs=1e-12)


def test_compute_cost_matches_matlab_analytic_shift() -> None:
    """RMSE parity against the MATLAB reference, computed analytically.

    For a constant per-frame shift ``s = [a, b, c]`` applied to both butt and
    clubhead, with quaternions identical:
        position_term     = 2 * (a^2 + b^2 + c^2)
        orientation_term  = 0
        anchor_term       = a^2 + b^2 + c^2          (clubhead at impact)
        regularizer       = 0 with lambda = 0
        J = w_position * 2 * |s|^2 + w_anchor_impact * |s|^2

    This is the formula MATLAB's ``compute_cost.m`` evaluates. Python output
    must match it to within 1e-6 RMSE per issue #4095.
    """
    target = _fixture_target()
    shift = np.array([0.01, -0.02, 0.03])
    sim = _shifted_sim(target, shift)
    opts = CostOptions(
        w_position=1.0,
        w_orientation=0.1,
        w_anchor_impact=10.0,
        lambda_=0.0,
        regularizer="coeff_l2",
    )
    s2 = float(np.dot(shift, shift))
    expected_position = opts.w_position * 2.0 * s2
    expected_anchor = opts.w_anchor_impact * s2
    expected_total = expected_position + expected_anchor

    theta = np.zeros(7)
    j, terms = compute_cost(theta, target, sim_fn=lambda _t: sim, opts=opts)

    rmse = np.sqrt((j - expected_total) ** 2)
    assert (
        rmse < PARITY_RMSE_TOL
    ), f"compute_cost RMSE {rmse:.3e} exceeds parity tolerance {PARITY_RMSE_TOL:.0e}"
    assert terms.position == pytest.approx(expected_position, abs=PARITY_RMSE_TOL)
    assert terms.orientation == pytest.approx(0.0, abs=PARITY_RMSE_TOL)
    assert terms.impact_anchor == pytest.approx(expected_anchor, abs=PARITY_RMSE_TOL)
    assert terms.regularizer == pytest.approx(0.0, abs=PARITY_RMSE_TOL)


def test_compute_total_work_matches_analytic() -> None:
    """``trapz(|tau * omega|)`` for piecewise-constant inputs is closed-form."""
    n = 11
    time = np.linspace(0.0, 1.0, n)
    tau = np.full((n, 2), 2.0)
    omega = np.full((n, 2), 3.0)
    sim = SimOutput(
        butt=np.zeros((n, 3)),
        clubhead=np.zeros((n, 3)),
        club_quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1)),
        time=time,
        tau=tau,
        omega=omega,
    )
    expected = 12.0  # int_0^1 (|2*3|+|2*3|) dt = 12
    actual = compute_total_work(sim)
    rmse = abs(actual - expected)
    assert rmse < PARITY_RMSE_TOL, f"compute_total_work RMSE {rmse:.3e}"


# NOTE: the original ``test_leaderboard_round_trip_assigns_ranks`` was
# removed during the rebase onto ``main``. The leaderboard API was
# rewritten by PR #4201 (issue #4097) into a JSON-row + Markdown report
# pipeline; ``build_leaderboard`` / ``rank_leaderboard`` no longer exist.
# Coverage of the new API lives in ``tests/unit/motion_matching/test_leaderboard.py``.
