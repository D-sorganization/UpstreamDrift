"""SHOW-C3c (#14559/#14561): ROM and velocity binding from rollout joint angles."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.kinematic_binding import (
    full_binding_report,
    kinematic_binding,
    kinematic_binding_report,
)
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.speed_continuation import (
    LadderSpec,
    run_speed_continuation,
)
from src.shared.python.impact_optimization.trajectory_optimiser import OptimiserConfig
from tests.unit.impact_optimization.test_trajectory_optimiser import (
    BASE,
    TIMES,
    _trust,
    plant,
)

pytestmark = pytest.mark.unit

NAMES = ("thoracic_axial", "NeckInputZ", "free")


def _env(rom=None, vel=2000.0) -> PhysiologicalEnvelope:
    rom_deg = np.array(
        rom or [(-50.0, 50.0), (-80.0, 80.0), (-np.inf, np.inf)], dtype=float
    )
    return PhysiologicalEnvelope(
        names=NAMES,
        tau_max_negative_nm=np.full(3, 500.0),
        tau_max_positive_nm=np.full(3, 500.0),
        rom_deg=rom_deg,
        vel_max_deg_s=np.full(3, vel),
        assumed_torque_slots=(),
        assumed_velocity_slots=(),
    )


T = np.linspace(0.0, 1.0, 101)


def _traj(thoracic=0.0, neck=0.0, free=0.0) -> np.ndarray:
    ramp = T / T[-1]
    return np.stack([thoracic * ramp, neck * ramp, free * ramp], axis=1)


def test_rom_binding_reports_peak_limit_and_overshoot_on_either_side() -> None:
    hits = kinematic_binding(_traj(thoracic=52.5, neck=-80.1), T, _env(), degrees=True)
    by = {(h.joint, h.kind): h for h in hits}
    th, nk = by[("thoracic_axial", "ROM")], by[("NeckInputZ", "ROM")]
    assert th.peak == pytest.approx(52.5) and th.limit == 50.0
    assert th.utilisation == pytest.approx(1.05)
    assert nk.peak == pytest.approx(-80.1) and nk.utilisation > 1.0


def test_inside_the_limits_nothing_binds_and_unbounded_rom_never_does() -> None:
    assert (
        kinematic_binding(
            _traj(thoracic=40.0, free=1e6), T, _env(vel=1e9), degrees=True
        )
        == ()
    )


def test_radians_are_converted_to_degrees() -> None:
    rad = _traj(thoracic=math.radians(52.5))
    hit = kinematic_binding(rad, T, _env(), degrees=False)[0]
    assert hit.peak == pytest.approx(52.5)


def test_velocity_binding_uses_the_rate_from_times() -> None:
    fast = _traj(neck=60.0)  # 60 deg in 1 s = 60 deg/s
    assert kinematic_binding(fast, T, _env(vel=100.0), degrees=True) == ()
    hit = kinematic_binding(fast, T, _env(vel=59.0), degrees=True)[0]
    assert (hit.joint, hit.kind) == ("NeckInputZ", "velocity")
    assert hit.peak == pytest.approx(60.0) and hit.utilisation > 1.0


def test_report_lines_carry_the_limit_kind_and_the_calibration_caveat() -> None:
    res = _ladder()
    env = _env([(-1.0, 1.0)] * 3)  # tiny ROM: reduced-plant angles must bind
    lines = kinematic_binding_report(
        res, env, TIMES, degrees=True, limit_notes={"thoracic_axial": "prior-derived"}
    )
    assert lines and all("ROM limit" in ln or "velocity limit" in ln for ln in lines)
    assert any("100 mph" in ln for ln in lines)
    assert any(
        "probable miscalibration" in ln and "not a measured" in ln for ln in lines
    )
    assert any("prior-derived" in ln for ln in lines if "thoracic_axial" in ln)


def test_full_report_has_torque_rom_and_velocity_lines() -> None:
    res = _ladder()
    env = _env([(-1.0, 1.0)] * 3, vel=1e-3)
    lines = full_binding_report(res, env, TIMES, degrees=True)
    kinds = {k for k in ("ROM limit", "velocity limit") if any(k in ln for ln in lines)}
    assert kinds == {"ROM limit", "velocity limit"}


def test_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ContractViolationError):
        kinematic_binding(_traj()[:, :2], T, _env(), degrees=True)
    with pytest.raises(ContractViolationError):
        kinematic_binding(_traj(), T[:-1], _env(), degrees=True)


def _ladder():
    return run_speed_continuation(
        rollout=plant,
        baseline_torque_nm=BASE,
        times_s=TIMES,
        envelope=_env(vel=2000.0),
        trust_region=_trust(2000.0),
        config=OptimiserConfig(n_knots=5, seed=0, max_iterations=20),
        ladder=LadderSpec(100.0, 105.0, 5.0),
    )
