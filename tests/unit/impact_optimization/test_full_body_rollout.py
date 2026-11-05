"""SHOW-C3c (#14559): full-body window rollout adapter for the impact optimiser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.full_body_rollout import (
    RolloutWindow,
    envelope_subset,
    make_full_body_rollout,
)
from src.shared.python.impact_optimization.physiological_envelope import (
    PhysiologicalEnvelope,
)
from src.shared.python.motion_matching.same_input.integrator import StepPolicy

pytestmark = pytest.mark.unit

ROOT = (
    "TranslationInputX", "TranslationInputY", "TranslationInputZ",
    "HipInputX", "HipInputY", "HipInputZ",
)  # fmt: skip
ORDER = (*ROOT, "J0", "J1", "J2")
DT = 0.01
STEPS = 20


@dataclass
class _Impact:
    face_deg: float | None
    path_deg: float | None
    aoa_deg: float
    speed_mph: float
    strike_offset_x_m: float | None = None
    strike_offset_y_m: float | None = None


class _Plant:
    """Unit-inertia joints: a = tau, root coordinates never accelerate."""

    coordinate_order = ORDER

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def acceleration(self, q, v, tau):
        self.seen.append(np.array(tau))
        a = np.array(tau, dtype=float)
        a[:6] = 0.0
        return a

    def closure_pose_residual(self, q):
        return np.zeros(1)


def _impact_fn(times, q, v) -> _Impact:
    return _Impact(face_deg=float(q[-1, 6]), path_deg=float(v[-1, 7]), aoa_deg=0.0,
                   speed_mph=float(np.abs(v[-1]).sum()))  # fmt: skip


def _window(plant=None, steps=STEPS) -> RolloutWindow:
    return RolloutWindow(
        plant=plant or _Plant(),
        q0=np.zeros(len(ORDER)),
        v0=np.zeros(len(ORDER)),
        baseline_efforts=np.tile(np.arange(len(ORDER), dtype=float), (steps, 1)),
        dt_s=DT,
        impact_fn=_impact_fn,
        policy=StepPolicy(substeps=1, project=False),
    )


def test_torque_columns_replace_the_baseline_and_root_stays_zero() -> None:
    plant = _Plant()
    win = _window(plant)
    rollout = make_full_body_rollout(win, ("J1", "J2"))
    tau = np.full((STEPS, 2), 7.0)
    rollout(tau)
    first = plant.seen[0]
    assert list(first[:6]) == [0.0] * 6  # root forced to zero
    assert first[6] == 6.0  # J0 keeps its baseline effort (index 6)
    assert list(first[7:]) == [7.0, 7.0]  # J1, J2 take the optimiser torque


def test_outcome_has_window_trajectory_in_joint_order_and_echoes_torque() -> None:
    win = _window()
    out = make_full_body_rollout(win, ("J2", "J0"))(np.zeros((STEPS, 2)))
    assert out.trajectory.shape == (STEPS, 2) and out.torque_nm.shape == (STEPS, 2)
    assert out.impact.face_deg == pytest.approx(float(out.trajectory[-1, 1]))  # J0 col
    assert win.times_s.shape == (STEPS,) and win.times_s[-1] == pytest.approx(
        STEPS * DT
    )


def test_baseline_torque_is_the_subset_of_the_baseline_efforts() -> None:
    win = _window()
    base = win.baseline_torque(("J1", "J0"))
    assert base.shape == (STEPS, 2) and list(base[0]) == [7.0, 6.0]


def test_rollout_is_deterministic() -> None:
    rollout = make_full_body_rollout(_window(), ("J0", "J1"))
    tau = np.random.default_rng(0).normal(size=(STEPS, 2))
    a, b = rollout(tau), rollout(tau)
    assert np.array_equal(a.trajectory, b.trajectory)


def test_bad_inputs_fail_closed() -> None:
    win = _window()
    with pytest.raises(ContractViolationError):
        make_full_body_rollout(win, ("HipInputX",))  # root is not controllable
    with pytest.raises(ContractViolationError):
        make_full_body_rollout(win, ("nope",))
    with pytest.raises(ContractViolationError):
        make_full_body_rollout(win, ("J0",))(np.zeros((STEPS + 1, 1)))


def _env() -> PhysiologicalEnvelope:
    n = 3
    return PhysiologicalEnvelope(
        names=("J0", "J1", "J2"),
        tau_max_negative_nm=np.array([10.0, 20.0, 30.0]),
        tau_max_positive_nm=np.array([11.0, 21.0, 31.0]),
        rom_deg=np.array([(-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0)]),
        vel_max_deg_s=np.array([100.0, 200.0, 300.0]),
        assumed_torque_slots=("J2",),
        assumed_velocity_slots=("J1",),
    )


def test_envelope_subset_keeps_limits_and_order_of_the_requested_joints() -> None:
    sub = envelope_subset(_env(), ("J2", "J0"))
    assert sub.names == ("J2", "J0")
    assert list(sub.tau_max_positive_nm) == [31.0, 11.0]
    assert list(sub.vel_max_deg_s) == [300.0, 100.0]
    assert sub.rom_deg.tolist() == [[-3.0, 3.0], [-1.0, 1.0]]
    assert sub.assumed_torque_slots == ("J2",) and sub.assumed_velocity_slots == ()
    with pytest.raises(ContractViolationError):
        envelope_subset(_env(), ("missing",))
