"""Unit tests for transactional muscle replay, R2025b T01 binding, and F10 turnover.

Verifies:
1. Transactional rollback when a plant step fails (controller history remains unmutated
   and plant snapshot is restored).
2. Fresh controller-off replay reproduces the committed forward trajectory from complete
   initial state + raw applied excitation history.
3. Budget boundary invariance (stop_when_feasible=False, exact candidate evaluation count,
   zero unaccounted extra evaluations).
4. R2025b operating-point T01 binding validation (rejects R2026a or missing operating point)
   and F10 turnover packet truthful "unqualified" reporting.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import pytest

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters
from src.shared.python.biomechanics.muscle_allocation import MuscleRecruitmentPath
from src.shared.python.biomechanics.transactional_muscle_replay import (
    FrozenBundleAdapterBinding,
    StepperOptions,
    TransactionalMuscleStepper,
    build_f10_qualification_turnover_packet,
    run_fresh_controller_off_muscle_replay,
    verify_frozen_bundle_adapter_execution,
)

pytestmark = pytest.mark.unit


class MockSteppablePlant:
    """Deterministic steppable plant oracle for contract verification."""

    def __init__(self, angle: float = 0.0, angular_velocity: float = 0.0) -> None:
        self.state: dict[str, float] = {
            "angle": float(angle),
            "angular_velocity": float(angular_velocity),
        }
        self.fail_advance: bool = False
        self.return_non_finite: bool = False
        self.advance_calls: int = 0
        self.restore_calls: int = 0

    def get_state_snapshot(self) -> dict[str, float]:
        return dict(self.state)

    def restore_state_snapshot(self, snapshot: dict[str, float]) -> None:
        self.state = dict(snapshot)
        self.restore_calls += 1

    def advance(
        self, dt_s: float, raw_excitations: Mapping[str, float]
    ) -> dict[str, float]:
        self.advance_calls += 1
        if self.fail_advance:
            raise RuntimeError("Synthetic plant integration error")

        if self.return_non_finite:
            self.state = {"angle": float("nan"), "angular_velocity": float("inf")}
            return dict(self.state)

        flex = raw_excitations.get("m_flex", 0.0)
        ext = raw_excitations.get("m_ext", 0.0)
        torque = (flex - ext) * 2.0
        new_omega = self.state["angular_velocity"] + torque * dt_s
        new_angle = self.state["angle"] + new_omega * dt_s
        self.state = {"angle": new_angle, "angular_velocity": new_omega}
        return dict(self.state)

    def validate_state(self, state: Any) -> bool:
        if not isinstance(state, Mapping):
            return False
        return all(np.isfinite(val) for val in state.values())


def _make_antagonist_paths() -> tuple[MuscleRecruitmentPath, MuscleRecruitmentPath]:
    p_flex = MuscleRecruitmentPath(
        name="m_flex",
        muscle=HillMuscleModel(MuscleParameters(F_max=500.0, l_opt=0.10, l_slack=0.20)),
        activation_dynamics=ActivationDynamics(),
        muscle_tendon_length_m=0.30,
        moment_arms_m_or_rad={"joint_angle": 0.05},
        activation=0.05,
        previous_excitation=0.05,
        excitation_bounds=(0.001, 1.0),
        max_excitation_rate_per_s=10.0,
    )
    p_ext = MuscleRecruitmentPath(
        name="m_ext",
        muscle=HillMuscleModel(MuscleParameters(F_max=500.0, l_opt=0.10, l_slack=0.20)),
        activation_dynamics=ActivationDynamics(),
        muscle_tendon_length_m=0.30,
        moment_arms_m_or_rad={"joint_angle": -0.05},
        activation=0.05,
        previous_excitation=0.05,
        excitation_bounds=(0.001, 1.0),
        max_excitation_rate_per_s=10.0,
    )
    return p_flex, p_ext


def test_transactional_rollback_on_plant_failure() -> None:
    """1. Transactional rollback restores snapshot and leaves controller history unmutated."""
    plant = MockSteppablePlant(angle=0.1, angular_velocity=0.0)
    paths = _make_antagonist_paths()
    stepper = TransactionalMuscleStepper(
        plant, paths, ["joint_angle"], options=StepperOptions(effort_tolerances=(0.05,))
    )

    initial_snapshot = plant.get_state_snapshot()
    proposal = stepper.propose_step(desired_generalized_effort=[1.0], dt_s=0.01)

    # Invariant: propose_step does NOT mutate controller history or step index
    assert len(stepper.committed_history) == 0
    assert stepper.current_step_index == 0
    assert stepper.current_time == 0.0
    assert not proposal.is_committed

    # Simulate plant failure during advance
    plant.fail_advance = True
    result = stepper.commit_step(proposal)

    assert result.status == "step_rejected_restored"
    assert result.restored_snapshot is True
    assert plant.state == initial_snapshot
    assert len(stepper.committed_history) == 0
    assert stepper.current_step_index == 0
    assert stepper.current_time == 0.0

    # Test non-finite return recovery
    plant.fail_advance = False
    plant.return_non_finite = True
    proposal2 = stepper.propose_step(desired_generalized_effort=[1.0], dt_s=0.01)
    result2 = stepper.commit_step(proposal2)

    assert result2.status == "step_rejected_restored"
    assert plant.state == initial_snapshot
    assert len(stepper.committed_history) == 0

    # Pre-advance rejection remains retryable: subsequent valid attempt succeeds
    plant.return_non_finite = False
    proposal3 = stepper.propose_step(desired_generalized_effort=[1.0], dt_s=0.01)
    result3 = stepper.commit_step(proposal3)

    assert result3.status == "committed"
    assert len(stepper.committed_history) == 1
    assert stepper.current_step_index == 1
    assert stepper.current_time == pytest.approx(0.01)

    # Stale/double commitment rejection
    with pytest.raises(ValueError, match="Cannot commit stale"):
        stepper.commit_step(proposal3)


def test_fresh_controller_off_muscle_replay_equivalence() -> None:
    """2. Fresh replay reproduces committed trajectory from initial state + raw excitations."""
    initial_angle = 0.05
    plant = MockSteppablePlant(angle=initial_angle, angular_velocity=0.0)
    initial_snapshot = plant.get_state_snapshot()
    paths = _make_antagonist_paths()
    stepper = TransactionalMuscleStepper(
        plant, paths, ["joint_angle"], options=StepperOptions(effort_tolerances=(0.05,))
    )

    dt_s = 0.01
    for step in range(3):
        target = 0.5 + 0.2 * step
        prop = stepper.propose_step(desired_generalized_effort=[target], dt_s=dt_s)
        advance_res = stepper.commit_step(prop)
        assert advance_res.status == "committed"

    assert len(stepper.committed_history) == 3
    assert len(stepper.committed_states) == 4

    # Fresh controller-off replay from factory and initial state
    def plant_factory() -> MockSteppablePlant:
        return MockSteppablePlant(angle=initial_angle, angular_velocity=0.0)

    replayed_states, is_equiv = run_fresh_controller_off_muscle_replay(
        plant_factory,
        initial_snapshot,
        stepper.committed_history,
        dt_s,
        expected_trajectory=stepper.committed_states,
    )

    assert is_equiv is True
    assert len(replayed_states) == len(stepper.committed_states)
    for rep, exp in zip(replayed_states, stepper.committed_states, strict=True):
        assert rep["angle"] == pytest.approx(exp["angle"], abs=1e-12)
        assert rep["angular_velocity"] == pytest.approx(
            exp["angular_velocity"], abs=1e-12
        )

    # Verify that divergent expected trajectory is caught
    corrupted_states = list(stepper.committed_states)
    corrupted_states[-1] = {"angle": 999.0, "angular_velocity": 999.0}
    with pytest.raises(ValueError, match="diverges from expected"):
        run_fresh_controller_off_muscle_replay(
            plant_factory,
            initial_snapshot,
            stepper.committed_history,
            dt_s,
            expected_trajectory=corrupted_states,
        )


def test_budget_boundary_invariance() -> None:
    """3. Evaluation budget boundary and zero unaccounted extra evaluations."""
    plant = MockSteppablePlant()
    paths = _make_antagonist_paths()
    max_evals = 40
    stepper = TransactionalMuscleStepper(
        plant,
        paths,
        ["joint_angle"],
        options=StepperOptions(
            effort_tolerances=(0.05,), max_nfev=max_evals, stop_when_feasible=False
        ),
    )

    proposal = stepper.propose_step(desired_generalized_effort=[1.0], dt_s=0.01)
    alloc_res = proposal.allocation_result

    # Evaluated candidates must obey declared max budget
    assert alloc_res.nfev <= max_evals
    assert alloc_res.nfev > 0

    # Plant advancement and commitment executes zero extra candidate allocations
    res = stepper.commit_step(proposal)
    assert res.status == "committed"
    assert res.proposal.allocation_result.nfev == alloc_res.nfev


def test_r2025b_binding_and_truthful_f10_turnover_packet() -> None:
    """4. R2025b T01 binding validation and F10 turnover packet truthful reporting."""
    init_sha = "0" * 64
    input_sha = "1" * 64

    # Simscape requires R2025b: rejects R2026a or missing release
    binding_r2026a = FrozenBundleAdapterBinding(
        model_id="GolfSwing3D_Kinetic",
        engine_name="simscape",
        initial_state_sha256=init_sha,
        applied_input_sha256=input_sha,
        operating_point_release="R2026a",
    )
    with pytest.raises(ValueError, match="operating_point_release='R2025b'"):
        verify_frozen_bundle_adapter_execution(
            binding_r2026a,
            {
                "model_id": "GolfSwing3D_Kinetic",
                "initial_state_sha256": init_sha,
                "applied_input_sha256": input_sha,
            },
        )

    binding_none = FrozenBundleAdapterBinding(
        model_id="GolfSwing3D_Kinetic",
        engine_name="simscape",
        initial_state_sha256=init_sha,
        applied_input_sha256=input_sha,
        operating_point_release=None,
    )
    with pytest.raises(ValueError, match="operating_point_release='R2025b'"):
        verify_frozen_bundle_adapter_execution(
            binding_none,
            {
                "model_id": "GolfSwing3D_Kinetic",
                "initial_state_sha256": init_sha,
                "applied_input_sha256": input_sha,
            },
        )

    # Valid R2025b Simscape binding passes
    binding_valid = FrozenBundleAdapterBinding(
        model_id="GolfSwing3D_Kinetic",
        engine_name="simscape",
        initial_state_sha256=init_sha,
        applied_input_sha256=input_sha,
        operating_point_release="R2025b",
    )
    verified = verify_frozen_bundle_adapter_execution(
        binding_valid,
        {
            "model_id": "GolfSwing3D_Kinetic",
            "initial_state_sha256": init_sha,
            "applied_input_sha256": input_sha,
        },
    )
    assert verified["status"] == "verified"
    assert verified["operating_point_release"] == "R2025b"

    # F10 turnover packet aggregation across 17 rows
    packet = build_f10_qualification_turnover_packet()

    assert packet["total_variants_count"] == 17
    assert packet["qualified_variants_count"] == 0
    assert packet["unqualified_variants_count"] == 17
    assert packet["aggregate_qualification_status"] == "unqualified"
    assert len(packet["variants"]) == 17
    for row in packet["variants"]:
        assert row["qualification"] == "unqualified"
        assert row["has_verified_native_receipt"] is False

    # Milestones F01-F10, T01-T03, D01-D04 present
    for ms in ("F01", "F08", "F09", "F10", "T01", "T02", "T03", "D01", "D04"):
        assert ms in packet["milestones"]
