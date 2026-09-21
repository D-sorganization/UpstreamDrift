"""Unit tests for double pendulum convention adapters and parity diagnostics (TB-04 #10589)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumParameters,
    DoublePendulumState,
    LowerSegmentProperties,
    SegmentProperties,
)
from src.shared.python.pendulum_simulator.physics import PendulumParams
from src.shared.python.simulation_backends.model_params import (
    GolfModelParams,
    LowerSegmentParams,
    UpperSegmentParams,
)
from src.shared.python.tour_baselines.pendulum_adapter import (
    analytical_to_tools_params,
    compare_dynamics_parity,
    golf_model_to_double_pendulum_params,
    state_to_planar_coordinates,
    tools_to_analytical_params,
)

pytestmark = pytest.mark.unit


def test_analytical_to_tools_params() -> None:
    """Verify analytical parameters map accurately to Tools simulator parameter format."""
    upper = SegmentProperties(
        length_m=0.62,
        mass_kg=1.8,
        center_of_mass_ratio=0.45,
        inertia_about_com=0.05,
    )
    lower = LowerSegmentProperties(
        length_m=0.88,
        shaft_mass_kg=0.12,
        clubhead_mass_kg=0.21,
        shaft_com_ratio=0.43,
    )
    analytical = DoublePendulumParameters(
        upper_segment=upper,
        lower_segment=lower,
        plane_inclination_deg=60.0,
        damping_shoulder=0.15,
        damping_wrist=0.08,
        gravity_m_s2=9.81,
        gravity_enabled=True,
    )

    tools = analytical_to_tools_params(analytical)

    assert pytest.approx(0.62) == tools.L1
    assert pytest.approx(0.88) == tools.L2
    assert tools.m1 == pytest.approx(1.8)
    assert tools.m2 == pytest.approx(0.12)
    assert tools.mClub == pytest.approx(0.21)
    assert tools.b1 == pytest.approx(0.15)
    assert tools.b2 == pytest.approx(0.08)
    # Projected gravity: g * cos(inclination) = 9.81 * cos(60 deg) = 4.905
    assert tools.g == pytest.approx(9.81 * math.cos(math.radians(60.0)))


def test_tools_to_analytical_params_distributed_and_point_mass() -> None:
    """Verify Tools params convert to analytical params under both approximations."""
    tools = PendulumParams(
        m1=1.5,
        m2=0.1,
        L1=0.6,
        L2=0.9,
        mClub=0.2,
        g=9.81,
        b1=0.2,
        b2=0.1,
        mu1=0.0,
        mu2=0.0,
    )

    # 1. Distributed inertia
    dist = tools_to_analytical_params(
        tools,
        plane_inclination_deg=45.0,
        arm_com_ratio=0.45,
        shaft_com_ratio=0.43,
        use_point_mass_approximation=False,
    )
    assert dist.upper_segment.length_m == 0.6
    assert dist.lower_segment.length_m == 0.9
    assert dist.upper_segment.mass_kg == 1.5
    assert dist.lower_segment.shaft_mass_kg == 0.1
    assert dist.lower_segment.clubhead_mass_kg == 0.2
    assert dist.damping_shoulder == 0.2
    assert dist.damping_wrist == 0.1
    assert dist.plane_inclination_deg == 45.0
    assert dist.upper_segment.center_of_mass_ratio == 0.45
    assert dist.upper_segment.inertia_about_com > 0.0

    # 2. Point mass approximation
    pt = tools_to_analytical_params(
        tools,
        plane_inclination_deg=0.0,
        use_point_mass_approximation=True,
    )
    assert pt.upper_segment.center_of_mass_ratio == 1.0
    assert pt.upper_segment.inertia_about_com == 0.0
    assert pt.lower_segment.shaft_mass_kg == 0.0
    assert pt.lower_segment.clubhead_mass_kg == pytest.approx(0.3)
    assert pt.lower_segment.total_mass == pytest.approx(0.3)


def test_state_to_planar_coordinates() -> None:
    """Verify coordinate and velocity vector extraction from DoublePendulumState."""
    state = DoublePendulumState(
        theta1=0.35,
        theta2=-1.20,
        omega1=2.5,
        omega2=-4.8,
    )
    q, v = state_to_planar_coordinates(state)

    assert q.shape == (2,)
    assert v.shape == (2,)
    assert np.allclose(q, [0.35, -1.20])
    assert np.allclose(v, [2.5, -4.8])


def test_golf_model_to_double_pendulum_params() -> None:
    """Verify conversion from GolfModelParams to DoublePendulumParameters."""
    model = GolfModelParams(
        upper=UpperSegmentParams(
            length_m=0.65,
            mass_kg=1.6,
            center_of_mass_ratio=0.45,
            inertia_about_com_kg_m2=0.04,
        ),
        lower=LowerSegmentParams(
            length_m=0.95,
            shaft_mass_kg=0.12,
            clubhead_mass_kg=0.20,
            shaft_com_ratio=0.43,
        ),
    )
    params = golf_model_to_double_pendulum_params(model)

    assert isinstance(params, DoublePendulumParameters)
    assert params.upper_segment.length_m == pytest.approx(0.65)
    assert params.upper_segment.mass_kg == pytest.approx(1.6)
    assert params.lower_segment.length_m == pytest.approx(0.95)


def test_compare_dynamics_parity_mathematical_equivalence() -> None:
    """Verify analytical point-mass formulation matches Tools dynamics to high precision."""
    tools = PendulumParams(
        m1=1.5,
        m2=0.15,
        L1=0.6,
        L2=0.85,
        mClub=0.22,
        g=9.81,
        b1=0.1,
        b2=0.05,
        mu1=0.0,
        mu2=0.0,
    )

    states = [
        (0.0, 0.0, 0.0, 0.0),
        (0.5, 1.0, 1.5, -2.0),
        (-1.2, 0.8, 5.0, 3.5),
        (2.1, -0.7, -4.0, 6.2),
    ]

    result = compare_dynamics_parity(tools, states, tolerance=1e-10)

    assert result.is_parity_acceptable
    assert result.mass_matrix_max_abs_error < 1e-10
    assert result.accel_max_abs_error < 1e-10
    assert "Parity verified" in result.notes
