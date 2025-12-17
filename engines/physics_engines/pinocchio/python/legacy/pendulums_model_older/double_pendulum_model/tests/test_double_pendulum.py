import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from double_pendulum_model.physics.double_pendulum import (  # noqa: E402
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    ExpressionFunction,
    LowerSegmentProperties,
    SegmentProperties,
)


def test_expression_function_allows_state_variables() -> None:
    expr = ExpressionFunction("0.5*sin(t) + 0.1*theta1 - 0.2*omega2")
    state = DoublePendulumState(theta1=0.2, theta2=-0.1, omega1=0.0, omega2=1.0)
    value = expr(1.0, state)
    expected = 0.5 * math.sin(1.0) + 0.1 * 0.2 - 0.2 * 1.0
    assert math.isclose(value, expected, rel_tol=1e-9)


def test_control_affine_matches_explicit_dynamics() -> None:
    control = (3.0, -1.0)
    parameters = DoublePendulumParameters.default()
    dynamics = DoublePendulumDynamics(
        parameters,
        forcing_functions=(lambda t, s: control[0], lambda t, s: control[1]),
    )
    state = DoublePendulumState(theta1=0.3, theta2=-0.4, omega1=0.2, omega2=-0.1)
    f, control_matrix = dynamics.control_affine(state)
    combined = (
        f[0] + control_matrix[0][0] * control[0] + control_matrix[0][1] * control[1],
        f[1] + control_matrix[1][0] * control[0] + control_matrix[1][1] * control[1],
        f[2] + control_matrix[2][0] * control[0] + control_matrix[2][1] * control[1],
        f[3] + control_matrix[3][0] * control[0] + control_matrix[3][1] * control[1],
    )
    derivatives = dynamics.derivatives(0.0, state)
    assert all(
        math.isclose(combined[i], derivatives[i], rel_tol=1e-9) for i in range(4)
    )


def test_joint_torque_breakdown_reports_components() -> None:
    parameters = DoublePendulumParameters.default()
    dynamics = DoublePendulumDynamics(parameters)
    state = DoublePendulumState(theta1=0.1, theta2=0.2, omega1=0.5, omega2=-0.3)
    torques: tuple[float, float] = (0.0, 0.0)
    breakdown = dynamics.joint_torque_breakdown(state, torques)
    assert breakdown.applied == torques
    assert breakdown.coriolis_centripetal != (0.0, 0.0)
    assert breakdown.gravitational[0] != 0.0
    assert breakdown.damping[0] == parameters.damping_shoulder * state.omega1


def test_gravity_projection_respects_plane_inclination() -> None:
    parameters = DoublePendulumParameters.default()
    projected = parameters.projected_gravity
    assert math.isclose(
        projected,
        parameters.gravity_m_s2
        * math.cos(math.radians(parameters.plane_inclination_deg)),
    )


def test_singular_mass_matrix_is_detected() -> None:
    upper_segment = SegmentProperties(
        length_m=0.0,
        mass_kg=1.0,
        center_of_mass_ratio=0.5,
        inertia_about_com=0.0,
    )
    lower_segment = LowerSegmentProperties(
        length_m=0.0, shaft_mass_kg=1.0, clubhead_mass_kg=1.0, shaft_com_ratio=0.5
    )
    parameters = DoublePendulumParameters(
        upper_segment=upper_segment, lower_segment=lower_segment
    )
    dynamics = DoublePendulumDynamics(parameters)
    state = DoublePendulumState(theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0)
    with pytest.raises(ZeroDivisionError):
        dynamics.control_affine(state)
