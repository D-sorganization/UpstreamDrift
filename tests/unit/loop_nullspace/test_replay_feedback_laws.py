"""Feedback laws, ground-load schedule and the provider closure hook (NSX-31)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.constrained_dynamics import kkt_projector
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    load_golden,
)
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    PlanarGolferProvider,
)
from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
from src.shared.python.loop_nullspace.replay import (
    AccelerationFeedbackLaw,
    FeedbackLaw,
    acceleration_feedback_law,
)
from src.shared.python.loop_nullspace.replay_ground import (
    GroundLoadSchedule,
    ground_forces_of_allocation,
)

pytestmark = pytest.mark.unit

N = 5


def _response(q: np.ndarray) -> np.ndarray:
    provider = PlanarGolferProvider(
        PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))
    )
    proj = kkt_projector(provider.mass_matrix(q), provider.closure_jacobian(q))
    return np.asarray(proj)


def _q() -> np.ndarray:
    return np.array(generic_planar_config().q)


def test_acceleration_law_commands_the_damped_spring_acceleration() -> None:
    q = _q()
    omega, zeta = 20.0, 0.7
    law = AccelerationFeedbackLaw(_response, N, omega, zeta)
    e = np.array([0.01, -0.02, 0.03, 0.0, 0.005])
    de = np.array([0.1, 0.0, -0.1, 0.2, 0.0])
    tau = law.torque(e, de, q)
    expected_acc = omega**2 * e + 2 * zeta * omega * de
    # the closure removes some directions: the commanded acceleration is met on
    # the controllable subspace (range of the response), the rest is dropped
    resp = _response(q)
    controllable = resp @ np.linalg.pinv(resp, hermitian=True) @ expected_acc
    np.testing.assert_allclose(resp @ tau, controllable, rtol=1e-6)


def test_acceleration_law_needs_the_configuration_and_validates() -> None:
    law = AccelerationFeedbackLaw(_response, N, 10.0)
    with pytest.raises(ValueError, match="configuration"):
        law.torque(np.zeros(N), np.zeros(N))
    with pytest.raises(ValueError, match="omega"):
        AccelerationFeedbackLaw(_response, N, 0.0)
    with pytest.raises(ValueError, match="zeta"):
        AccelerationFeedbackLaw(_response, N, 1.0, zeta=-1.0)
    with pytest.raises(ValueError, match="rcond"):
        AccelerationFeedbackLaw(_response, N, 1.0, rcond=2.0)
    with pytest.raises(TypeError, match="callable"):
        AccelerationFeedbackLaw(np.eye(N), N, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="torque_limit"):
        AccelerationFeedbackLaw(_response, N, 1.0, torque_limit=np.zeros(N))


def test_acceleration_law_saturates_at_the_declared_limit() -> None:
    law = AccelerationFeedbackLaw(_response, N, 50.0, torque_limit=np.full(N, 0.5))
    tau = law.torque(np.full(N, 1.0), np.zeros(N), _q())
    assert np.max(np.abs(tau)) == pytest.approx(0.5)


def test_frozen_matrix_law_matches_the_state_dependent_law_at_its_point() -> None:
    q = _q()
    frozen = acceleration_feedback_law(_response(q), 20.0, 1.0)
    live = AccelerationFeedbackLaw(_response, N, 20.0, 1.0)
    e, de = np.linspace(-0.01, 0.01, N), np.linspace(0.1, -0.1, N)
    np.testing.assert_allclose(
        frozen.torque(e, de), live.torque(e, de, q), rtol=1e-6, atol=1e-9
    )
    with pytest.raises(ValueError, match="square"):
        acceleration_feedback_law(np.zeros((2, 3)), 1.0)
    with pytest.raises(ValueError, match="omega"):
        acceleration_feedback_law(np.eye(2), -1.0)


def test_matrix_gain_law_must_be_symmetric_psd() -> None:
    FeedbackLaw(kp=np.eye(2), kd=np.eye(2))
    with pytest.raises(ValueError, match="symmetric"):
        FeedbackLaw(kp=np.array([[1.0, 2.0], [0.0, 1.0]]), kd=np.eye(2))
    with pytest.raises(ValueError, match="PSD"):
        FeedbackLaw(kp=-np.eye(2), kd=np.eye(2))
    with pytest.raises(ValueError, match="shape"):
        FeedbackLaw(kp=np.ones(2), kd=np.ones(3))


class _FakeAdapter:
    nv = 4
    contact_names = ("a", "b")

    def compute_contact_jacobian(self, q: np.ndarray) -> np.ndarray:
        jac = np.zeros((6, 4))
        jac[:3, :3] = np.eye(3)  # contact a drives coordinates 0..2
        jac[3:, 1:] = 2.0 * np.eye(3)  # contact b drives coordinates 1..3
        return jac


def test_ground_load_is_the_transposed_jacobian_times_force() -> None:
    adapter = _FakeAdapter()
    sched = GroundLoadSchedule(
        adapter,
        [0.0, 1.0],
        [("a",), ("a",)],
        [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])],
    )
    np.testing.assert_allclose(sched(0.0, np.zeros(4)), [1.0, 2.0, 3.0, 0.0])
    np.testing.assert_allclose(sched(0.5, np.zeros(4)), [2.0, 2.0, 2.0, 0.0])
    np.testing.assert_allclose(sched(9.0, np.zeros(4)), [3.0, 2.0, 1.0, 0.0])


def test_ground_load_holds_forces_across_a_contact_set_change() -> None:
    sched = GroundLoadSchedule(
        _FakeAdapter(),
        [0.0, 1.0],
        [("a",), ("b",)],
        [np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0])],
    )
    np.testing.assert_allclose(sched(0.9, np.zeros(4)), [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_allclose(sched(1.0, np.zeros(4)), [0.0, 2.0, 0.0, 0.0])


def test_ground_load_with_no_contacts_is_zero_and_preconditions_hold() -> None:
    adapter = _FakeAdapter()
    sched = GroundLoadSchedule(adapter, [0.0], [()], [np.zeros(0)])
    np.testing.assert_array_equal(sched(0.0, np.zeros(4)), np.zeros(4))
    with pytest.raises(ValueError, match="increasing"):
        GroundLoadSchedule(adapter, [1.0, 0.0], [("a",)] * 2, [np.zeros(3)] * 2)
    with pytest.raises(ValueError, match="one contact"):
        GroundLoadSchedule(adapter, [0.0, 1.0], [("a",)], [np.zeros(3)])
    with pytest.raises(ValueError, match="unknown"):
        GroundLoadSchedule(adapter, [0.0], [("zz",)], [np.zeros(3)])
    with pytest.raises(ValueError, match="length 3"):
        GroundLoadSchedule(adapter, [0.0], [("a",)], [np.zeros(2)])


def test_ground_forces_are_the_last_block_of_x() -> None:
    x = [np.arange(10.0), np.arange(10.0) + 100.0]
    forces = ground_forces_of_allocation(x, n_actuated=4, n_multipliers=3)
    np.testing.assert_array_equal(forces[0], [7.0, 8.0, 9.0])
    np.testing.assert_array_equal(forces[1], [107.0, 108.0, 109.0])


def test_provider_exposes_closure_error_only_when_the_adapter_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.loop_nullspace import provider as provider_module

    monkeypatch.setattr(
        provider_module, "require_closed_loop_capability", lambda a: None
    )

    class _Plain:
        nv = 2

    class _WithError(_Plain):
        def closure_position_error(self, q: np.ndarray) -> np.ndarray:
            return 2.0 * np.asarray(q)

    assert getattr(AdapterConstrainedProvider(_Plain()), "closure_error", None) is None
    prov = AdapterConstrainedProvider(_WithError())
    np.testing.assert_allclose(prov.closure_error(np.array([1.0, 2.0])), [2.0, 4.0])
    with pytest.raises(ValueError, match="shape"):
        prov.closure_error(np.zeros(3))
