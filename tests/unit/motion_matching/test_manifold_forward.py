"""Independent order checks for local-chart RK4 on configurations and tangents."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.manifold_forward import (
    integrate_manifold_forward,
)

pytestmark = pytest.mark.unit


def _so3_integrate(q, u):
    return (Rotation.from_quat(q) * Rotation.from_rotvec(u.copy())).as_quat()


def _so3_difference_rate(anchor, q, velocity):
    u = (Rotation.from_quat(anchor).inv() * Rotation.from_quat(q)).as_rotvec()
    theta = np.linalg.norm(u)
    x, y, z = u
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    coefficient = (
        1 / 12 if theta < 1e-6 else (1 - theta / (2 * np.tan(theta / 2))) / theta**2
    )
    return (np.eye(3) + 0.5 * skew + coefficient * skew @ skew) @ velocity


def test_noncommuting_rotation_has_fourth_order_convergence():
    # Independent exact R(t)=Rx(t) Ry(t^2), whose angular velocity in the
    # moving body is [cos(t^2), 2t, sin(t^2)]. Rotations do not commute.
    target = Rotation.from_rotvec([1, 0, 0]) * Rotation.from_rotvec([0, 1, 0])
    errors = []
    for step in (0.2, 0.1, 0.05):
        result = integrate_manifold_forward(
            np.array([0, 0, 0, 1.0]),
            np.array([1.0, 0, 0]),
            np.array([0.0, 1.0]),
            lambda t, q, v: np.array(
                [-2 * t * np.sin(t * t), 2.0, 2 * t * np.cos(t * t)]
            ),
            integrate=_so3_integrate,
            difference_rate=_so3_difference_rate,
            max_step=step,
        )
        error = np.linalg.norm(
            (target.inv() * Rotation.from_quat(result.configuration[-1])).as_rotvec()
        )
        errors.append(error)
        assert result.configuration.shape == (2, 4)
        assert result.velocity.shape == (2, 3)
        np.testing.assert_allclose(
            np.linalg.norm(result.configuration, axis=1), 1, atol=2e-15
        )
        assert result.evaluations == 4 * result.steps
    assert 12 < errors[0] / errors[1] < 20
    assert 12 < errors[1] / errors[2] < 20
    assert errors[-1] < 3e-7


def test_euclidean_limit_exact_clock_and_owned_readonly_outputs():
    q0, v0, clock = np.array([0.0]), np.array([2.0]), np.array([0.0, 0.13, 0.31, 0.7])
    calls = []

    def acceleration(t, q, v):
        calls.append(t)
        return np.array([3.0])

    result = integrate_manifold_forward(
        q0,
        v0,
        clock,
        acceleration,
        integrate=lambda q, u: q + u,
        difference_rate=lambda anchor, q, v: v,
        max_step=0.05,
    )
    np.testing.assert_array_equal(result.time, clock)
    np.testing.assert_allclose(
        result.configuration[:, 0], 2 * clock + 1.5 * clock**2, atol=2e-15
    )
    np.testing.assert_allclose(result.velocity[:, 0], 2 + 3 * clock, atol=2e-15)
    assert min(calls) == 0
    assert max(calls) == pytest.approx(clock[-1])
    q0[:], v0[:], clock[:] = 9, 9, 9
    assert result.time[0] == result.configuration[0, 0] == 0
    assert result.velocity[0, 0] == 2
    for array in (result.time, result.configuration, result.velocity):
        assert not array.flags.writeable


@pytest.mark.parametrize(
    "clock,step",
    [
        (np.array([0.0, 0.0]), 0.1),
        (np.array([1.0, 2.0]), 0.1),
        (np.array([0.0, 1.0]), 0),
        (np.array([0.0, 1.0]), np.nan),
    ],
)
def test_invalid_clock_or_step_is_rejected(clock, step):
    with pytest.raises(ValueError):
        integrate_manifold_forward(
            np.ones(1),
            np.ones(1),
            clock,
            lambda t, q, v: v,
            integrate=lambda q, u: q + u,
            difference_rate=lambda a, q, v: v,
            max_step=step,
        )


@pytest.mark.parametrize("kind", ["acceleration", "integrate", "difference_rate"])
def test_malformed_callback_output_is_rejected(kind):
    callbacks = {
        "acceleration": lambda t, q, v: v,
        "integrate": lambda q, u: q + u,
        "difference_rate": lambda a, q, v: v,
    }
    callbacks[kind] = lambda *args: np.array([np.nan])
    with pytest.raises(ValueError, match=kind):
        integrate_manifold_forward(
            np.ones(1), np.ones(1), np.array([0.0, 0.1]), **callbacks
        )


def test_euclidean_coupled_configuration_velocity_has_fourth_order():
    errors = []
    for h in (0.2, 0.1, 0.05):
        result = integrate_manifold_forward(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0, 1.0]),
            lambda t, q, v: -q,
            integrate=lambda q, u: q + u,
            difference_rate=lambda anchor, q, v: v,
            max_step=h,
        )
        errors.append(
            np.linalg.norm(
                [
                    result.configuration[-1, 0] - np.cos(1),
                    result.velocity[-1, 0] + np.sin(1),
                ]
            )
        )
    assert 14 < errors[0] / errors[1] < 18
    assert 14 < errors[1] / errors[2] < 18


def test_adaptive_tighter_tolerances_reduce_oscillator_error():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    errors, costs = [], []
    for tolerance in (1e-4, 1e-7, 1e-10):
        result = integrate_manifold_adaptive(
            np.array([1.0]),
            np.array([0.0]),
            np.array([0.0, 0.37, 1.0]),
            lambda t, q, v: -9 * q,
            integrate=lambda q, u: q + u,
            difference_rate=lambda a, q, v: v,
            difference=lambda a, q: q - a,
            rtol=tolerance,
            atol=tolerance,
            max_step=0.5,
        )
        errors.append(
            np.linalg.norm(
                [
                    result.configuration[-1, 0] - np.cos(3),
                    result.velocity[-1, 0] + 3 * np.sin(3),
                ]
            )
        )
        costs.append(result.evaluations)
        np.testing.assert_array_equal(result.time, [0.0, 0.37, 1.0])
        assert result.evaluations % 12 == 0
        assert result.evaluations >= 12 * result.steps
        assert not result.configuration.flags.writeable
    assert errors[1] < errors[0] / 50
    assert errors[2] < errors[1] / 50
    assert costs[0] < costs[1] < costs[2]
    assert errors[-1] < 2e-8


def test_adaptive_noncommuting_rotation_matches_independent_solution():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    result = integrate_manifold_adaptive(
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0]),
        lambda t, q, v: np.array([-2 * t * np.sin(t * t), 2.0, 2 * t * np.cos(t * t)]),
        integrate=_so3_integrate,
        difference_rate=_so3_difference_rate,
        difference=lambda a, q: (
            Rotation.from_quat(a).inv() * Rotation.from_quat(q)
        ).as_rotvec(),
        rtol=1e-9,
        atol=1e-11,
        max_step=0.2,
    )
    target = Rotation.from_rotvec([1, 0, 0]) * Rotation.from_rotvec([0, 1, 0])
    assert (
        np.linalg.norm(
            (target.inv() * Rotation.from_quat(result.configuration[-1])).as_rotvec()
        )
        < 2e-8
    )
    np.testing.assert_allclose(
        result.velocity[-1], [np.cos(1), 2, np.sin(1)], atol=2e-8
    )


def test_adaptive_budget_fails_before_exceeding_rhs_limit():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    evaluations = []

    def acceleration(t, q, v):
        evaluations.append(t)
        return -q

    with pytest.raises(RuntimeError, match="evaluation budget"):
        integrate_manifold_adaptive(
            np.ones(1),
            np.zeros(1),
            np.array([0.0, 1.0]),
            acceleration,
            integrate=lambda q, u: q + u,
            difference_rate=lambda a, q, v: v,
            difference=lambda a, q: q - a,
            max_step=0.1,
            max_evaluations=13,
        )
    assert len(evaluations) == 12


@pytest.mark.parametrize(
    "option,value",
    [
        ("rtol", 0),
        ("atol", np.nan),
        ("max_evaluations", True),
        ("max_evaluations", 2.5),
    ],
)
def test_adaptive_rejects_invalid_controls(option, value):
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    with pytest.raises(ValueError):
        integrate_manifold_adaptive(
            np.ones(1),
            np.zeros(1),
            np.array([0.0, 1.0]),
            lambda t, q, v: -q,
            integrate=lambda q, u: q + u,
            difference_rate=lambda a, q, v: v,
            difference=lambda a, q: q - a,
            **{option: value},
        )


def test_adaptive_underflow_and_bad_difference_fail_explicitly():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    callbacks = {
        "integrate": lambda q, u: q + u,
        "difference_rate": lambda a, q, v: v,
        "difference": lambda a, q: q - a,
    }
    with pytest.raises(RuntimeError, match="underflow"):
        integrate_manifold_adaptive(
            np.ones(1),
            np.zeros(1),
            np.array([0.0, np.nextafter(0.0, 1.0)]),
            lambda t, q, v: -q,
            **callbacks,
        )
    callbacks["difference"] = lambda a, q: np.array([np.nan])
    with pytest.raises(ValueError, match="difference"):
        integrate_manifold_adaptive(
            np.ones(1),
            np.zeros(1),
            np.array([0.0, 0.1]),
            lambda t, q, v: -q,
            **callbacks,
        )


def test_adaptive_lands_on_boundary_without_one_ulp_residual_step():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    boundary = 1.0 / 30.0
    nearly_boundary = np.nextafter(boundary, 0.0)
    assert nearly_boundary == 0.033333333333333326
    evaluations = []

    def acceleration(t, q, v):
        evaluations.append(t)
        return np.zeros(1)

    result = integrate_manifold_adaptive(
        np.zeros(1),
        np.ones(1),
        np.array([0.0, boundary]),
        acceleration,
        integrate=lambda q, u: q + u,
        difference_rate=lambda a, q, v: v,
        difference=lambda a, q: q - a,
        max_step=nearly_boundary,
    )
    assert result.steps == 1
    assert result.evaluations == 12
    assert evaluations[-1] == boundary
    np.testing.assert_allclose(result.configuration[-1], [boundary], rtol=0, atol=1e-17)


def test_adaptive_genuine_tiny_requested_step_still_underflows():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_adaptive,
    )

    with pytest.raises(RuntimeError, match="underflow"):
        integrate_manifold_adaptive(
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 1.0]),
            lambda t, q, v: np.zeros(1),
            integrate=lambda q, u: q + u,
            difference_rate=lambda a, q, v: v,
            difference=lambda a, q: q - a,
            max_step=np.nextafter(0.0, 1.0),
        )


def test_dop853_noncommuting_rotation_and_absolute_clock():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_dop853,
    )

    clock = np.array([0.0, 0.13, 0.37, 1.0])
    evaluations = []

    def acceleration(t, q, v):
        evaluations.append(t)
        return np.array([-2 * t * np.sin(t * t), 2.0, 2 * t * np.cos(t * t)])

    result = integrate_manifold_dop853(
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        clock,
        acceleration,
        integrate=_so3_integrate,
        difference_rate=_so3_difference_rate,
        max_step=0.1,
        rtol=1e-10,
        atol=1e-12,
    )
    for i, t in enumerate(clock):
        expected = Rotation.from_rotvec([t, 0, 0]) * Rotation.from_rotvec([0, t * t, 0])
        assert (
            np.linalg.norm(
                (
                    expected.inv() * Rotation.from_quat(result.configuration[i])
                ).as_rotvec()
            )
            < 1e-10
        )
        np.testing.assert_allclose(
            result.velocity[i], [np.cos(t * t), 2 * t, np.sin(t * t)], atol=1e-10
        )
    assert min(evaluations) == 0
    assert max(evaluations) == 1
    assert result.evaluations == len(evaluations)
    np.testing.assert_array_equal(result.time, clock)


def test_dop853_euclidean_continuation_preserves_physical_endpoint():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_dop853,
    )

    clock = np.array([0.0, 0.11, 0.37, 1.0])
    result = integrate_manifold_dop853(
        np.array([0.3]),
        np.array([-0.2]),
        clock,
        lambda t, q, v: np.array([t]),
        integrate=lambda q, u: q + u,
        difference_rate=lambda a, q, v: v,
        max_step=0.1,
    )
    np.testing.assert_allclose(
        result.configuration[:, 0], 0.3 - 0.2 * clock + clock**3 / 6, atol=2e-14
    )
    np.testing.assert_allclose(result.velocity[:, 0], -0.2 + clock**2 / 2, atol=2e-14)
    assert result.steps >= len(clock) - 1
    for array in (result.configuration, result.velocity, result.time):
        assert not array.flags.writeable


def test_dop853_evaluation_budget_is_global_across_intervals():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_dop853,
    )

    callbacks = {
        "integrate": lambda q, u: q + u,
        "difference_rate": lambda a, q, v: v,
        "max_step": 0.1,
    }
    first = integrate_manifold_dop853(
        np.zeros(1),
        np.ones(1),
        np.array([0.0, 0.1]),
        lambda t, q, v: np.zeros(1),
        **callbacks,
    )
    calls = []

    def acceleration(t, q, v):
        calls.append(t)
        return np.zeros(1)

    with pytest.raises(RuntimeError, match="evaluation budget"):
        integrate_manifold_dop853(
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 0.1, 0.2]),
            acceleration,
            max_evaluations=first.evaluations + 3,
            **callbacks,
        )
    assert len(calls) == first.evaluations + 3
    assert max(calls) > 0.1


@pytest.mark.parametrize(
    "options", [{"rtol": 0}, {"atol": np.nan}, {"max_evaluations": True}]
)
def test_dop853_invalid_controls(options):
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_dop853,
    )

    with pytest.raises(ValueError):
        integrate_manifold_dop853(
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 0.1]),
            lambda t, q, v: v,
            integrate=lambda q, u: q + u,
            difference_rate=lambda a, q, v: v,
            **options,
        )


def test_dop853_malformed_callback_fails_without_partial_result():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_dop853,
    )

    with pytest.raises(ValueError, match="acceleration"):
        integrate_manifold_dop853(
            np.zeros(1),
            np.ones(1),
            np.array([0.0, 0.1]),
            lambda t, q, v: np.array([np.nan]),
            integrate=lambda q, u: q + u,
            difference_rate=lambda a, q, v: v,
        )


def test_dop853_actual_pinocchio_noncommuting_motion():
    from src.shared.python.motion_matching.manifold_forward import (
        integrate_manifold_dop853,
    )

    pin = pytest.importorskip("pinocchio")
    if not isinstance(getattr(pin, "__version__", None), str):
        pytest.skip("Real Pinocchio runtime required")
    model = pin.Model()
    model.addJoint(0, pin.JointModelSpherical(), pin.SE3.Identity(), "rotation")

    def difference_rate(anchor, q, v):
        return pin.dDifference(model, anchor, q, pin.ARG1) @ v

    result = integrate_manifold_dop853(
        pin.neutral(model),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.17, 0.61, 1.0]),
        lambda t, q, v: np.array([-2 * t * np.sin(t * t), 2.0, 2 * t * np.cos(t * t)]),
        integrate=lambda q, u: pin.integrate(model, q, u),
        difference_rate=difference_rate,
        max_step=0.1,
        rtol=1e-10,
        atol=1e-12,
    )
    expected = Rotation.from_rotvec([1, 0, 0]) * Rotation.from_rotvec([0, 1, 0])
    assert (
        np.linalg.norm(
            (expected.inv() * Rotation.from_quat(result.configuration[-1])).as_rotvec()
        )
        < 1e-10
    )
