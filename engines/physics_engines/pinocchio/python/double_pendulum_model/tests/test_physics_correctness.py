import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from double_pendulum_model.physics.double_pendulum import (  # noqa: E402
    DoublePendulumDynamics,
    DoublePendulumParameters,
    LowerSegmentProperties,
    SegmentProperties,
)
from double_pendulum_model.physics.triple_pendulum import (  # noqa: E402
    TriplePendulumDynamics,
    TriplePendulumState,
)


def test_double_pendulum_physics_values() -> None:
    """
        Verify mass matrix and coriolis terms against analytical derivation for
    specific parameters.
    Parameters:
    m1=1, l1=1, lc1=0.5, Ic1=1/12 -> I_proximal_1 = 1/3
    m2=1, l2=1, lc2=0.5, Ic2=1/12 -> I_proximal_2 = 1/3
    """
    upper = SegmentProperties(
        length_m=1.0,
        mass_kg=1.0,
        center_of_mass_ratio=0.5,
        inertia_about_com=1.0 / 12.0,
    )
    lower = LowerSegmentProperties(
        length_m=1.0, shaft_mass_kg=1.0, clubhead_mass_kg=0.0, shaft_com_ratio=0.5
    )
    params = DoublePendulumParameters(
        upper_segment=upper, lower_segment=lower, gravity_enabled=False
    )
    dynamics = DoublePendulumDynamics(parameters=params)

    # Check Mass Matrix at theta2 = 0
    # Expected: M11 = 8/3, M22 = 1/3, M12 = 5/6
    mass_matrix = dynamics.mass_matrix(theta2=0.0)

    assert math.isclose(mass_matrix[0][0], 8.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(mass_matrix[1][1], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(mass_matrix[0][1], 5.0 / 6.0, rel_tol=1e-9)

    # Check Coriolis at theta2 = pi/2, omega1 = 1.0
    # Expected C2 = m2 * l1 * lc2 * sin(theta2) * omega1^2 = 1 * 1 * 0.5 * 1 * 1 = 0.5

    _c1, c2 = dynamics.coriolis_vector(theta2=math.pi / 2.0, omega1=1.0, omega2=0.0)
    assert math.isclose(c2, 0.5, rel_tol=1e-9)


def test_triple_pendulum_stability_at_zero() -> None:
    """
    Verify that theta=0 is a stable equilibrium (restoring gravity).
    """
    dynamics = TriplePendulumDynamics()  # Uses default parameters

    # Small displacement
    state = TriplePendulumState(
        theta1=0.1, theta2=0.0, theta3=0.0, omega1=0.0, omega2=0.0, omega3=0.0
    )

    # Get gravity torque (bias vector with zero velocity)
    # bias = C + G. With omega=0, C=0. Bias = G.
    # Forward dynamics: M acc = - G (since control=0)
    # If stable, acc should be negative (restoring).

    acc1, _acc2, _acc3 = dynamics.forward_dynamics(state, control=(0.0, 0.0, 0.0))

    # Since we displaced theta1 positively, we expect acc1 to be negative
    assert acc1 < 0.0, f"Triple pendulum unstable at theta=0! acc1={acc1}"
