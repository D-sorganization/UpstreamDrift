# fmt: off
"""Analytic oracle for the two-hand null-space load family (#15731, #15494)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.hand_load_nullspace.simulation_output import (
    SCHEMA_VERSION, Channel, cop_channel, simulation_output, superposition_holds, zvcf_from_pinned,
)
from src.shared.python.hand_load_nullspace.solver import (
    HandLoadProblem, certify_forward_replay, component_extrema, grasp_null_residual,
    kkt_residual, solve, synthetic_forward_step, time_weighted_squared_force,
)

pytestmark = pytest.mark.unit
_FORCE = (10.0, 0.0, 4.0)


def _problem(**overrides) -> HandLoadProblem:
    base = {
        "p_l": (0.0, 0.0, 0.0), "p_r": (0.0, 0.0, 0.2), "force_n": _FORCE,
        "moment_nm": (0.0, 2.0, 0.0), "couple_policy": "fixed",
    }
    base.update(overrides)
    return HandLoadProblem(**base)


def test_analytic_oracle_family_and_min_squared_force():
    family, best = solve(_problem(alpha=0.5)), solve(_problem(objective="min_squared_force"))
    assert family.f_l_n == pytest.approx((-5.0, 0.0, 2.5))
    assert family.f_r_n == pytest.approx((15.0, 0.0, 1.5))
    assert family.objective == "prescribed_alpha" and best.objective == "min_squared_force"
    assert best.alpha == pytest.approx(0.0) and best.squared_force < family.squared_force
    assert grasp_null_residual((0.0, 0.0, 0.0), (0.0, 0.0, 0.2)) == pytest.approx(0.0)


def test_min_cancellation_beats_its_neighbours():
    best = solve(_problem(objective="min_cancellation"))
    got = np.linalg.norm(best.f_l_n) + np.linalg.norm(best.f_r_n) - np.linalg.norm(_FORCE)
    assert best.opposing_force_cancellation == pytest.approx(got)
    for step in (-0.5, 0.5):
        other = solve(_problem(alpha=best.alpha + step))
        assert other.opposing_force_cancellation >= best.opposing_force_cancellation - 1e-9


def test_kkt_residuals_vanish_only_at_the_optima():
    for name in ("min_squared_force", "min_cancellation"):
        residual = kkt_residual(_problem(objective=name), solve(_problem(objective=name)))
        assert residual.availability == "available"
        assert residual.primal_force == pytest.approx(0.0, abs=1e-8)
        assert residual.primal_moment == pytest.approx(0.0, abs=1e-8)
        assert residual.stationarity == pytest.approx(0.0, abs=1e-8)
    assert kkt_residual(_problem(), solve(_problem(alpha=0.4))).stationarity > 1e-3


def test_infeasible_axial_moment_is_unavailable_not_zero_filled():
    sol = solve(_problem(moment_nm=(0.0, 0.0, 1.0)))
    assert sol.feasible is False and sol.availability == "UNAVAILABLE"
    assert sol.f_l_n is None and sol.f_r_n is None
    assert sol.squared_force is None and sol.opposing_force_cancellation is None
    assert "u · m'" in sol.reason
    assert time_weighted_squared_force((sol,), 0.01) is None


def test_reference_point_covariance_preserves_hand_forces():
    at_mid = solve(_problem())
    moved = solve(_problem(moment_nm=(0.0, 3.0, 0.0), reference_point_m=(0.0, 0.0, 0.0)))
    assert moved.f_l_n == pytest.approx(at_mid.f_l_n)
    assert moved.f_r_n == pytest.approx(at_mid.f_r_n)
    assert moved.midhands_moment_nm == pytest.approx(at_mid.midhands_moment_nm)


def test_component_extrema_move_only_along_the_grip_axis():
    report = component_extrema(_problem(), -1.0, 1.0)
    width = dict(zip(report.names, report.width, strict=True))
    assert report.availability == "available"
    assert width["f_l_x"] == pytest.approx(0.0, abs=1e-9) and width["f_l_z"] == pytest.approx(2.0)
    assert width["F_x"] == pytest.approx(0.0, abs=1e-9) and width["M_y"] == pytest.approx(0.0, abs=1e-9)
    blocked = component_extrema(_problem(moment_nm=(0.0, 0.0, 1.0)), -1.0, 1.0)
    assert blocked.availability == "UNAVAILABLE" and blocked.minimum == ()


def test_forward_replay_matches_the_net_wrench_and_ignores_alpha():
    base, shifted = solve(_problem(alpha=0.0)), solve(_problem(alpha=1.5))
    mass, inertia, dt = 0.2, np.diag([0.01, 0.01, 0.001]), 0.01
    zeros = (0.0, 0.0, 0.0)
    step = synthetic_forward_step(zeros, zeros, zeros, base.net_force_n, base.midhands_moment_nm, mass, inertia, dt)
    other = synthetic_forward_step(zeros, zeros, zeros, shifted.net_force_n, shifted.midhands_moment_nm, mass, inertia, dt)
    assert np.allclose(step[0], other[0]) and np.allclose(step[2], other[2])
    kwargs = {"mass": mass, "inertia": inertia, "position": zeros, "velocity": zeros, "omega": zeros, "dt": dt, "reference_velocity": step[1], "reference_omega": step[2]}
    assert base.replay_badge == "allocation-only"
    assert certify_forward_replay(base, reference_position=step[0], **kwargs).replay_badge == "forward-replayed"
    assert certify_forward_replay(base, reference_position=(1.0, 0.0, 0.0), **kwargs).replay_badge == "allocation-only"


def test_couple_policy_and_total_only_grip_check():
    axial = solve(_problem(moment_nm=(0.0, 0.0, 1.0), couple_policy="axial_only"))
    assert axial.m_l_nm == pytest.approx((0.0, 0.0, 0.5)) and axial.m_r_nm == pytest.approx((0.0, 0.0, 0.5))
    slack = solve(_problem(couple_policy="bounded", couple_bound_nm=2.0))
    assert slack.f_l_n == pytest.approx(tuple(0.5 * c for c in _FORCE))
    tight = solve(_problem(moment_nm=(0.0, 0.0, 1.0), couple_policy="bounded", couple_bound_nm=0.1))
    assert tight.availability == "UNAVAILABLE"
    total = solve(_problem(max_force_n=1.0))
    zvcf = solve(_problem(max_force_n=1.0, load_kind="zvcf", objective="min_cancellation"))
    assert total.grip_checked and total.f_l_n is None and total.availability == "UNAVAILABLE"
    assert zvcf.feasible and zvcf.grip_checked is False and zvcf.load_kind == "zvcf"


def test_pinned_attribution_and_simulation_output_contract():
    low, high = solve(_problem(alpha=0.0)), solve(_problem(alpha=2.0))
    assert low.net_force_n == pytest.approx(high.net_force_n)
    assert low.midhands_moment_nm == pytest.approx(high.midhands_moment_nm)
    ztcf_f = tuple(0.25 * c for c in low.net_force_n)
    ztcf_m = tuple(0.25 * c for c in low.midhands_moment_nm)
    zvcf_f, zvcf_m = zvcf_from_pinned(low.net_force_n, low.midhands_moment_nm, ztcf_f, ztcf_m)
    again_f, again_m = zvcf_from_pinned(high.net_force_n, high.midhands_moment_nm, ztcf_f, ztcf_m)
    assert np.allclose(zvcf_f, again_f) and np.allclose(zvcf_m, again_m)
    assert superposition_holds(low.net_force_n, ztcf_f, zvcf_f, low.midhands_moment_nm, ztcf_m, zvcf_m)
    assert not superposition_holds(low.net_force_n, ztcf_f, (6.0, 0.0, 3.0), low.midhands_moment_nm, ztcf_m, zvcf_m)
    out = simulation_output(
        _problem(), low, frame="world", time_s=0.2, ground_force_n=(0.0, 0.0, 4.0),
        ground_moment_nm=(0.0, 0.0, 0.0), ground_point_m=(0.0, 0.0, 0.0), match_id="match-1",
    )
    assert out.schema_version == SCHEMA_VERSION and out.replay_badge == "allocation-only"
    assert out.cop_m.availability == "UNAVAILABLE" and out.cop_m.values is None
    assert out.ground_force_n.values == pytest.approx((0.0, 0.0, 4.0))
    assert cop_channel((0.0, 0.0, 20.0), (0.0, -2.0, 0.0), (0.0, 0.0, 0.0)).values == pytest.approx((0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="zero-fill|must not carry values"):
        Channel("UNAVAILABLE", "N", (0.0, 0.0, 0.0), "missing")


def test_preconditions_fail_closed():
    with pytest.raises(ValueError, match="distinct"):
        solve(_problem(p_r=(0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="finite"):
        solve(_problem(force_n=(1.0, float("nan"), 0.0)))
# fmt: on
