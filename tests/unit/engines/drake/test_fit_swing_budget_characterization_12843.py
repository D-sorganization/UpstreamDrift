"""Characterization tests pinning ``fit_swing_drake`` before the #12843 split."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.drake.python.motion_matching import fit_swing as fs
from src.engines.physics_engines.drake.python.motion_matching.fit_swing import (
    FitOptions,
    fit_swing_drake,
)
from src.engines.physics_engines.drake.python.motion_matching.simulate import SimOut
from src.shared.python.motion_matching.club_target import ClubTarget
from tests.engines.drake.test_drake_fit_swing import _make_target


def _sim_for(target: ClubTarget):  # type: ignore[no-untyped-def]
    def _sim(theta: np.ndarray) -> SimOut:
        off = theta[:3].reshape(1, 3)
        n = target.time.shape[0]
        return SimOut(
            time=target.time,
            q=np.zeros((n, 1)),
            qd=np.zeros((n, 1)),
            qdd=np.zeros((n, 1)),
            tau=np.zeros((n, 1)),
            grip=target.butt + off,
            grip_quat=target.club_quat.copy(),
            clubhead=target.clubhead + off,
            club_quat=target.club_quat.copy(),
        )

    return _sim


@pytest.mark.unit
def test_fit_result_fields_and_provenance_placeholders() -> None:
    target = _make_target()
    res = fit_swing_drake(
        target, FitOptions(n_joints=1, max_iterations=5), simulate_fn=_sim_for(target)
    )
    assert res.theta_optimal.shape == (7,)
    assert res.method == "SLSQP"
    assert res.n_evaluations == len(res.history) > 0
    assert res.solver_status in {"success", "warning", "failed"}
    assert res.git_commit == "unknown"
    assert res.engine_version == "unknown"
    assert res.target_hash == "unknown"
    assert res.timestamp_utc == "unknown"
    assert res.wall_clock_s >= 0.0
    assert np.isfinite(res.final_rmse_m)


@pytest.mark.unit
def test_fit_warm_start_is_clipped_to_bounds() -> None:
    target = _make_target()
    seen: list[np.ndarray] = []
    inner = _sim_for(target)

    def _spy(theta: np.ndarray) -> SimOut:
        seen.append(theta.copy())
        return inner(theta)

    theta0 = np.full(7, 1e6)
    fit_swing_drake(
        target,
        FitOptions(n_joints=1, theta0=theta0, max_iterations=1),
        simulate_fn=_spy,
    )
    lb, ub = fs.polynomial_parameter_bounds(1)
    np.testing.assert_allclose(seen[0], ub)
    assert np.all(seen[0] >= lb)


@pytest.mark.unit
def test_fit_theta0_wrong_length_and_bad_joints() -> None:
    target = _make_target()
    with pytest.raises(ValueError, match="theta0 has length 3, expected 7"):
        fit_swing_drake(target, FitOptions(n_joints=1, theta0=np.zeros(3)))
    with pytest.raises(ValueError, match="n_joints must be positive"):
        fit_swing_drake(target, FitOptions(n_joints=0))
