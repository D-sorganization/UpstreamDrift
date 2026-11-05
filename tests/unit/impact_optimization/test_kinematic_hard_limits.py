"""SHOW-C3c D29 (#14561): ROM and velocity are HARD constraints on a ladder rung.

A rung that meets the impact tolerances but drives a joint past its range of
motion or rate limit is not feasible.  It must not raise ``reached_mph``, and
the report must name the joint, the limit and the value as a probable
miscalibration (125 mph is firm), not silently accept the over-range rollout.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.kinematic_binding import (
    enforce_kinematic_limits,
    full_binding_report,
    kinematic_binding,
    require_kinematically_feasible,
)
from src.shared.python.impact_optimization.speed_continuation import (
    LadderSpec,
    run_speed_continuation,
)
from src.shared.python.impact_optimization.trajectory_optimiser import OptimiserConfig
from tests.unit.impact_optimization.test_trajectory_optimiser import (
    BASE,
    TIMES,
    _envelope,
    _trust,
    plant,
)

pytestmark = pytest.mark.unit


def _ladder():
    return run_speed_continuation(
        plant,
        BASE,
        TIMES,
        _envelope(),
        _trust(2000.0),
        OptimiserConfig(n_knots=5, seed=0, max_iterations=40),
        ladder=LadderSpec(100.0, 105.0, 5.0),
    )


def _peak_deg(result) -> float:
    return max(
        float(np.degrees(np.abs(p.trajectory[:, 0])).max()) for p in result.points
    )


def _with_rom(half_range_deg: float):
    lo_hi = np.tile([-np.inf, np.inf], (3, 1))
    lo_hi[0] = (-half_range_deg, half_range_deg)
    return dataclasses.replace(
        _envelope(), rom_deg=lo_hi, vel_max_deg_s=np.full(3, 1e12)
    )


def test_a_rung_inside_every_limit_keeps_its_feasibility() -> None:
    result = _ladder()
    assert all(p.feasible for p in result.points)
    out, kin = enforce_kinematic_limits(result, _with_rom(1e6), TIMES)
    assert out == result
    assert all(h.utilisation <= 1.0 for hits in kin for h in hits)


def test_rom_overshoot_makes_an_otherwise_feasible_rung_infeasible() -> None:
    result = _ladder()
    env = _with_rom(0.5 * _peak_deg(result))
    out, kin = enforce_kinematic_limits(result, env, TIMES)
    assert not any(p.feasible for p in out.points)
    assert out.reached_mph == 0.0
    assert any(
        h.kind == "ROM" and h.joint == "J0" and h.utilisation > 1.0 for h in kin[0]
    )
    assert [p.solution for p in out.points] == [p.solution for p in result.points]


def test_velocity_overshoot_is_also_a_hard_constraint() -> None:
    result = _ladder()
    env = dataclasses.replace(_with_rom(1e6), vel_max_deg_s=np.full(3, 1e-3))
    out, _ = enforce_kinematic_limits(result, env, TIMES)
    assert not any(p.feasible for p in out.points)


def test_reached_mph_drops_only_the_violating_rungs() -> None:
    result = _ladder()
    peaks = [float(np.degrees(np.abs(p.trajectory[:, 0])).max()) for p in result.points]
    assert max(peaks) - min(peaks) > 1e-6, "plant must give rungs distinct peaks"
    out, _ = enforce_kinematic_limits(
        result, _with_rom((min(peaks) + max(peaks)) / 2), TIMES
    )
    expected = [pk <= (min(peaks) + max(peaks)) / 2 for pk in peaks]
    assert [p.feasible for p in out.points] == expected
    assert out.reached_mph == max(
        (p.speed_target_mph for p in out.points if p.feasible), default=0.0
    )


def test_report_names_joint_limit_value_and_probable_miscalibration() -> None:
    result = _ladder()
    env = _with_rom(0.5 * _peak_deg(result))
    out, _ = enforce_kinematic_limits(result, env, TIMES)
    text = "\n".join(full_binding_report(out, env, TIMES))
    assert "J0" in text and "ROM limit" in text
    assert "BLOCKS" in text and "probable miscalibration" in text


def test_postcondition_rejects_a_feasible_rung_beyond_a_limit() -> None:
    result = _ladder()
    env = _with_rom(0.5 * _peak_deg(result))
    kin = tuple(
        kinematic_binding(p.trajectory, TIMES, env, degrees=False)
        for p in result.points
    )
    with pytest.raises(ContractViolationError):
        require_kinematically_feasible(result, kin)
