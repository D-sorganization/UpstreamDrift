"""F02 controller behavior on independent analytic and native plants."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.local_policy import (
    LinearizedDynamics,
    tvlqr_gains,
)
from src.shared.python.motion_matching.distributed_feedback import (
    ActuatorMap,
    ContactConstrainedAllocator,
    ControlTask,
    DirectBoundedAllocator,
    DistributedFeedbackController,
    EuclideanTangentModel,
    NominalTrajectory,
    TaskGroup,
    TaskKinematics,
)
from src.shared.python.motion_matching.contact_force_allocator import (
    ContactForceAllocation,
    ContactForceAllocator,
    FeasibilityStatus,
)

pytestmark = pytest.mark.unit


class LinearTask:
    def __init__(self, jacobian: np.ndarray) -> None:
        self.jacobian = np.asarray(jacobian, dtype=float)

    def sample(self, q: np.ndarray, v: np.ndarray) -> TaskKinematics:
        return TaskKinematics(self.jacobian @ q, self.jacobian @ v, self.jacobian)

    def difference(self, target: np.ndarray, actual: np.ndarray) -> np.ndarray:
        return target - actual


def _fixture(
    *,
    steps: int = 80,
    q_ref: float = 0.0,
    gain_sign: float = 1.0,
    enabled: bool = True,
) -> DistributedFeedbackController:
    dt = 0.02
    a = np.array([[1.0, dt], [0.0, 1.0]])
    b = np.array([[0.0], [dt]])
    gains = tvlqr_gains(
        LinearizedDynamics(np.tile(a, (steps, 1, 1)), np.tile(b, (steps, 1, 1))),
        np.diag([10.0, 1.0]),
        np.array([[0.1]]),
    )
    schedule = NominalTrajectory(
        times=np.linspace(0.0, steps * dt, steps + 1),
        q=np.full((steps, 1), q_ref),
        v=np.zeros((steps, 1)),
        feedforward=np.zeros((steps, 1)),
        gains=gain_sign * gains,
        phase_names=("downswing",) * steps,
        channel_ids=("hinge",),
    )
    actuators = ActuatorMap(("hinge",), (0,), ())
    return DistributedFeedbackController(
        EuclideanTangentModel(1),
        actuators,
        schedule,
        DirectBoundedAllocator(
            actuators,
            np.array([-20.0]),
            np.array([20.0]),
            np.array([500.0]),
            contact_free=True,
        ),
        enabled=enabled,
    )


def _rollout(
    controller: DistributedFeedbackController, start: float = 0.4
) -> np.ndarray:
    q, v = np.array([start]), np.array([0.0])
    dt = 0.02
    for i in range(80):
        step = controller.command_for_step(i * dt, q, v, dt)
        assert step.information_pattern == "exact_simulated_state"
        v = v + step.applied[0] * dt
        q = q + v * dt
    return q


def test_correct_tvlqr_sign_restores_and_reversed_sign_fails() -> None:
    correct = _rollout(_fixture())
    reversed_sign = _rollout(_fixture(gain_sign=-1.0))
    assert abs(correct[0]) < 0.1
    assert abs(reversed_sign[0]) > 0.4


def test_disabled_feedback_exposes_frozen_feedforward_only() -> None:
    controller = _fixture(enabled=False)
    step = controller.command_for_step(0.0, np.array([0.4]), np.array([0.0]), 0.02)
    np.testing.assert_array_equal(step.nominal_feedforward, [0.0])
    np.testing.assert_array_equal(step.feedback_correction, [0.0])
    np.testing.assert_array_equal(step.total_requested, [0.0])
    np.testing.assert_array_equal(step.applied, [0.0])


def test_rate_and_saturation_apply_once_at_actuator_boundary() -> None:
    controller = _fixture()
    first = controller.command_for_step(0.0, np.array([10.0]), np.array([0.0]), 0.02)
    second = controller.command_for_step(0.02, np.array([-10.0]), np.array([0.0]), 0.02)
    assert abs(first.applied[0]) <= 20.0
    assert abs(second.applied[0]) <= 20.0
    assert abs(second.applied[0] - first.applied[0]) <= 10.0 + 1e-12
    assert first.total_requested[0] != first.applied[0]


def test_phase_and_priority_keep_lower_conflicting_task_out_of_higher_axis() -> None:
    base = _fixture(steps=2, enabled=True)
    schedule = replace(
        base.schedule,
        times=np.array([0.0, 0.1, 0.2]),
        phase_names=("address", "downswing"),
        gains=np.zeros((2, 1, 2)),
    )
    high = ControlTask(
        "pelvis",
        TaskGroup.PELVIS_FEET,
        0,
        ("address",),
        LinearTask(np.array([[1.0]])),
        np.array([[1.0], [1.0]]),
        np.zeros((2, 1)),
        kp=10.0,
        kd=0.0,
        frame_id="world",
        position_unit="rad",
    )
    low = replace(
        high,
        name="club",
        group=TaskGroup.CLUB,
        priority=1,
        active_phases=("address", "downswing"),
        target_positions=np.array([[-1.0], [-1.0]]),
    )
    controller = DistributedFeedbackController(
        base.model, base.actuators, schedule, base.allocator, tasks=(high, low)
    )
    address = controller.command_for_step(0.0, np.array([0.0]), np.array([0.0]), 0.1)
    downswing = controller.command_for_step(0.1, np.array([0.0]), np.array([0.0]), 0.1)
    assert address.feedback_correction[0] > 0.0
    assert address.active_tasks == ("pelvis", "club")
    assert downswing.feedback_correction[0] < 0.0
    assert downswing.active_tasks == ("club",)


def test_task_provider_uses_shortest_manifold_angle_error() -> None:
    class WrappedAngleTask(LinearTask):
        def difference(self, target: np.ndarray, actual: np.ndarray) -> np.ndarray:
            raw = target - actual
            return np.arctan2(np.sin(raw), np.cos(raw))

    base = _fixture(steps=1)
    task = ControlTask(
        "wrist",
        TaskGroup.WRISTS_HANDS,
        0,
        ("downswing",),
        WrappedAngleTask(np.array([[1.0]])),
        np.array([[3.13]]),
        np.zeros((1, 1)),
        kp=10.0,
        kd=0.0,
        frame_id="wrist",
        position_unit="rad",
    )
    controller = DistributedFeedbackController(
        base.model, base.actuators, base.schedule, base.allocator, tasks=(task,)
    )
    step = controller.command_for_step(0.0, np.array([-3.13]), np.zeros(1), 0.02)
    assert -0.3 < step.feedback_correction[0] < 0.0


def test_manifold_difference_uses_nv_tangent_not_nq_coordinates() -> None:
    class QuaternionLike:
        nq = 4
        nv = 3

        def difference(self, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
            return q[1:] - reference[1:]

        def mass_inverse(self, q: np.ndarray) -> np.ndarray:
            return np.eye(3)

    schedule = NominalTrajectory(
        np.array([0.0, 0.1]),
        np.array([[1.0, 0.0, 0.0, 0.0]]),
        np.zeros((1, 3)),
        np.zeros((1, 1)),
        np.ones((1, 1, 6)),
        ("address",),
        ("joint",),
    )
    act = ActuatorMap(("joint",), (2,), (0, 1))
    controller = DistributedFeedbackController(
        QuaternionLike(),
        act,
        schedule,
        DirectBoundedAllocator(
            act, np.array([-2.0]), np.array([2.0]), np.array([100.0]), contact_free=True
        ),
    )
    step = controller.command_for_step(
        0.0, np.array([1.0, 0.0, 0.0, 0.1]), np.zeros(3), 0.1
    )
    assert step.feedback_correction[0] < 0.0
    with pytest.raises(ValueError, match="nq=nv"):
        EuclideanTangentModel(3, nq=4)


def test_actuator_permutation_and_hidden_root_drive_fail_closed() -> None:
    controller = _fixture(steps=2)
    wrong_order = replace(controller.schedule, channel_ids=("other",))
    with pytest.raises(ValueError, match="channel order"):
        DistributedFeedbackController(
            controller.model,
            controller.actuators,
            wrong_order,
            controller.allocator,
        )
    with pytest.raises(ValueError, match="root"):
        ActuatorMap(("root",), (0,), (0,))
    with pytest.raises(ValueError, match="contact-free"):
        DirectBoundedAllocator(
            controller.actuators,
            np.array([-1.0]),
            np.array([1.0]),
            np.array([1.0]),
            contact_free=False,
        )


def test_control_samples_are_contiguous_and_do_not_cross_policy_boundary() -> None:
    controller = _fixture(steps=2)
    with pytest.raises(ValueError, match="boundary"):
        controller.command_for_step(0.0, np.zeros(1), np.zeros(1), 0.03)
    controller.command_for_step(0.0, np.zeros(1), np.zeros(1), 0.01)
    with pytest.raises(ValueError, match="contiguous"):
        controller.command_for_step(0.015, np.zeros(1), np.zeros(1), 0.005)


def test_missing_task_manifold_capability_fails_closed() -> None:
    base = _fixture(steps=1)
    task = ControlTask(
        "club",
        TaskGroup.CLUB,
        0,
        ("downswing",),
        object(),
        np.zeros((1, 1)),
        np.zeros((1, 1)),
        kp=1.0,
        kd=0.0,
        frame_id="clubhead",
        position_unit="m",
    )
    with pytest.raises(ValueError, match="adapter lacks"):
        DistributedFeedbackController(
            base.model, base.actuators, base.schedule, base.allocator, tasks=(task,)
        )


def test_all_five_distributed_groups_have_explicit_phase_activation() -> None:
    base = _fixture(steps=1)
    tasks = tuple(
        ControlTask(
            group.value,
            group,
            rank,
            ("downswing",),
            LinearTask(np.array([[1.0]])),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            kp=1.0,
            kd=0.0,
            frame_id=group.value,
            position_unit="rad",
        )
        for rank, group in enumerate(TaskGroup)
    )
    controller = DistributedFeedbackController(
        base.model, base.actuators, base.schedule, base.allocator, tasks=tasks
    )
    step = controller.command_for_step(0.0, np.array([0.1]), np.zeros(1), 0.02)
    assert step.active_tasks == tuple(group.value for group in TaskGroup)
    assert np.isfinite(step.applied).all()


def test_invalid_mass_inverse_fails_before_allocation() -> None:
    class IndefiniteModel:
        nq = 1
        nv = 1

        def difference(self, q: np.ndarray, reference: np.ndarray) -> np.ndarray:
            return q - reference

        def mass_inverse(self, q: np.ndarray) -> np.ndarray:
            return np.array([[-1.0]])

    base = _fixture(steps=1)
    controller = DistributedFeedbackController(
        IndefiniteModel(), base.actuators, base.schedule, base.allocator
    )
    with pytest.raises(ValueError, match="positive definite"):
        controller.command_for_step(0.0, np.zeros(1), np.zeros(1), 0.02)


def test_contact_allocator_rejects_hidden_reserve_and_permutation() -> None:
    class Geometry:
        own_contact = True

        def required_generalized_force(
            self, q: np.ndarray, v: np.ndarray, requested: np.ndarray
        ) -> np.ndarray:
            return requested.copy()

        def linearization(self, q: np.ndarray, v: np.ndarray):
            return (
                np.zeros((3, 7)),
                np.zeros((6, 7)),
                np.array([False]),
                np.array([0.0, 0.0, 1.0]),
            )

    class Kernel:
        nv = 7
        actuated_indices = np.array([6])

        def allocate(self, requested: np.ndarray, *args, **kwargs):
            return ContactForceAllocation(
                tau_actuated=np.array([requested[6]]),
                f_ground=np.zeros(3),
                lambda_grip=np.zeros(6),
                delta_tau_root=np.array([1.0, 0, 0, 0, 0, 0]),
                equilibrium_residual=0.0,
                root_balance_residual=1.0,
                success=True,
                is_physically_feasible=True,
                feasibility_status=FeasibilityStatus.FEASIBLE,
                root_slack_norm=1.0,
            )

    actuators = ActuatorMap(("joint",), (6,), tuple(range(6)))
    with pytest.raises(ValueError, match="permutation"):
        ContactConstrainedAllocator(
            ActuatorMap(("joint",), (7,), tuple(range(6))),
            Kernel(),
            Geometry(),
            np.array([-10.0]),
            np.array([10.0]),
            np.array([100.0]),
        )
    allocator = ContactConstrainedAllocator(
        actuators,
        Kernel(),
        Geometry(),
        np.array([-10.0]),
        np.array([10.0]),
        np.array([100.0]),
    )
    with pytest.raises(ValueError, match="hidden root reserve"):
        allocator.allocate(np.zeros(7), None, 0.01, np.zeros(7), np.zeros(7))


def test_contact_allocator_uses_existing_qp_on_feasible_torque_case() -> None:
    class Geometry:
        own_contact = True

        def required_generalized_force(
            self, q: np.ndarray, v: np.ndarray, requested: np.ndarray
        ) -> np.ndarray:
            return requested.copy()

        def linearization(self, q: np.ndarray, v: np.ndarray):
            return (
                np.zeros((3, 7)),
                np.zeros((6, 7)),
                np.array([False]),
                np.array([0.0, 0.0, 1.0]),
            )

    actuators = ActuatorMap(("joint",), (6,), tuple(range(6)))
    allocator = ContactConstrainedAllocator(
        actuators,
        ContactForceAllocator(7, [6], n_contact_spheres=1),
        Geometry(),
        np.array([-10.0]),
        np.array([10.0]),
        np.array([100.0]),
    )
    desired = np.zeros(7)
    desired[6] = 2.0
    receipt = allocator.allocate(desired, None, 0.01, np.zeros(7), np.zeros(7))
    assert receipt.contact_mode == "native_own_contact_qp_prediction"
    assert receipt.root_slack_norm < 1e-8
    assert receipt.predicted_ground is not None
    assert receipt.predicted_grip is not None
    np.testing.assert_allclose(receipt.applied, [2.0], atol=1e-4)


def test_contact_allocator_requires_native_own_contact_dynamics() -> None:
    class Geometry:
        own_contact = False

        def linearization(self, q: np.ndarray, v: np.ndarray):
            return (
                np.zeros((3, 7)),
                np.zeros((6, 7)),
                np.array([False]),
                np.array([0.0, 0.0, 1.0]),
            )

    actuators = ActuatorMap(("joint",), (6,), tuple(range(6)))
    with pytest.raises(ValueError, match="native own-contact dynamics"):
        ContactConstrainedAllocator(
            actuators,
            ContactForceAllocator(7, [6], n_contact_spheres=1),
            Geometry(),
            np.array([-10.0]),
            np.array([10.0]),
            np.array([100.0]),
        )


def test_nonunit_or_state_dependent_transmission_fails_closed() -> None:
    with pytest.raises(ValueError, match="transmission"):
        ActuatorMap(("joint",), (0,), (), transmission_type="state_dependent")


def test_native_mujoco_torque_step_has_no_prescribed_coordinates() -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = """<mujoco><worldbody><body name='arm'><joint name='hinge' type='hinge'/>
    <geom type='capsule' size='.02 .1' mass='1'/></body></worldbody>
    <actuator><motor name='hinge' joint='hinge' gear='1'/></actuator></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = 0.2
    controller = _fixture()
    first = controller.command_for_step(
        0.0, data.qpos.copy(), data.qvel.copy(), model.opt.timestep
    )
    assert first.applied[0] < 0.0
    data.ctrl[0] = first.applied[0]
    mujoco.mj_step(model, data)
    assert data.qpos[0] != 0.2
    assert np.isfinite(data.qpos).all()
