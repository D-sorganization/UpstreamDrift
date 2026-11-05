"""SHOW-C3c slice 3a (#14559): guarded full-body speed ladder (no real run)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.full_body_ladder import (
    BaselineBand,
    held_joints,
    make_mujoco_impact_fn,
    require_physical_baseline,
    run_full_body_ladder,
)
from src.shared.python.impact_optimization.full_body_rollout import RolloutWindow
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.impact_optimization.speed_continuation import LadderSpec
from src.shared.python.impact_optimization.trajectory_optimiser import OptimiserConfig
from src.shared.python.motion_matching.same_input.integrator import StepPolicy

pytestmark = pytest.mark.unit

ROOT = (
    "TranslationInputX", "TranslationInputY", "TranslationInputZ",
    "HipInputX", "HipInputY", "HipInputZ",
)  # fmt: skip
ORDER = (*ROOT, "J0", "J1", "J2")
DT = 0.01
STEPS = 8


@dataclass
class _Impact:
    face_deg: float | None
    path_deg: float | None
    aoa_deg: float
    speed_mph: float
    strike_offset_x_m: float | None = 0.0
    strike_offset_y_m: float | None = 0.0


class _Plant:
    coordinate_order = ORDER

    def acceleration(self, q, v, tau):
        a = np.array(tau, dtype=float)
        a[:6] = 0.0
        return a

    def closure_pose_residual(self, q):
        return np.zeros(1)


def _window(base_speed_gain: float = 100.0) -> RolloutWindow:
    def impact_fn(times, q, v):
        # speed grows with J0 velocity at the end of the window
        return _Impact(0.0, 0.0, 0.0, base_speed_gain + 40.0 * float(v[-1, 6]))

    eff = np.zeros((STEPS, len(ORDER)))
    eff[:, 6] = 1.0
    return RolloutWindow(
        plant=_Plant(),
        q0=np.zeros(len(ORDER)),
        v0=np.zeros(len(ORDER)),
        baseline_efforts=eff,
        dt_s=DT,
        impact_fn=impact_fn,
        policy=StepPolicy(substeps=1, project=False),
    )


def _env() -> PhysiologicalEnvelope:
    return PhysiologicalEnvelope(
        names=("J0", "J1", "J2"),
        tau_max_negative_nm=np.full(3, 50.0),
        tau_max_positive_nm=np.full(3, 50.0),
        rom_deg=np.array([(-90.0, 90.0)] * 3),
        vel_max_deg_s=np.full(3, 1.0e4),
        assumed_torque_slots=(),
        assumed_velocity_slots=(),
    )


def test_band_accepts_a_physical_baseline_and_rejects_the_rest() -> None:
    band = BaselineBand()
    require_physical_baseline(_Impact(10.0, 3.0, 2.0, 110.0), band)
    for bad in (
        _Impact(10.0, 3.0, 2.0, 80.0),  # too slow (unrefit spec)
        _Impact(-66.0, 3.0, 2.0, 110.0),  # face far off
        _Impact(10.0, -142.0, 2.0, 110.0),  # path far off
        _Impact(None, 3.0, 2.0, 110.0),  # unavailable is never invented
    ):
        with pytest.raises(ContractViolationError):
            require_physical_baseline(bad, band)


def test_band_validates_its_own_bounds() -> None:
    with pytest.raises(ContractViolationError):
        BaselineBand(speed_mph_min=120.0, speed_mph_max=100.0)
    with pytest.raises(ContractViolationError):
        BaselineBand(max_abs_face_deg=0.0)


def test_ladder_refuses_to_start_on_a_non_physical_baseline() -> None:
    with pytest.raises(ContractViolationError, match="baseline"):
        run_full_body_ladder(
            _window(base_speed_gain=50.0),
            ("J0", "J1"),
            _env(),
            band=BaselineBand(),
            trust_radius=1.0,
        )


def test_ladder_runs_on_a_physical_baseline_and_reports_per_rung_bindings() -> None:
    run = run_full_body_ladder(
        _window(),
        ("J0", "J1"),
        _env(),
        band=BaselineBand(),
        trust_radius=5.0,
        config=OptimiserConfig(n_knots=2, max_iterations=3),
        ladder=LadderSpec(100.0, 105.0, 5.0),
    )
    assert len(run.result.points) == 2
    assert len(run.kinematic_by_rung) == 2
    assert run.baseline.speed_mph == pytest.approx(100.0 + 40.0 * 0.08)
    assert run.controlled == ("J0", "J1") and run.held == ("J2",)


def test_held_joints_are_the_non_root_complement_in_plant_order() -> None:
    assert held_joints(_window(), ("J1",)) == ("J0", "J2")


def test_mujoco_impact_fn_scatters_by_joint_id_and_evaluates_at_the_index() -> None:
    class _Model:
        nq, nv = 4, 4
        jnt_qposadr = np.array([2, 0, 3])
        jnt_dofadr = np.array([2, 0, 3])

    seen = {}

    def fake_eval(model, rollout, body, frame, *, spec, impact_index):
        seen.update(q=rollout.qpos, v=rollout.qvel, idx=impact_index, body=body)
        return _Impact(1.0, 2.0, 3.0, 4.0)

    fn = make_mujoco_impact_fn(
        _Model(), [0, 1, 2], "club", "frame", "face", evaluate=fake_eval
    )
    t = np.array([0.0, 0.1])
    q = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = fn(t, q, q * 10)
    assert out.speed_mph == 4.0 and seen["idx"] == 1 and seen["body"] == "club"
    assert seen["q"][1].tolist() == [5.0, 0.0, 4.0, 6.0]
