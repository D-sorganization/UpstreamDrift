"""Robertson–Winter interface power (#15390)."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.swing_model import PlanarInertials
from src.shared.python.biomechanics.robertson_winter_interface_power import (
    endpoint_interface_power_w,
    segment_energy_balance_residual_w,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def _golf_params():
    from src.shared.python.simulation_backends import GolfModelParams

    return GolfModelParams.default()


@pytest.fixture
def params(_golf_params):
    return _golf_params


@pytest.fixture
def inertials(params):
    return PlanarInertials.from_params(params)


def test_endpoint_power_sign_and_units() -> None:
    """Positive force along joint translation does positive work."""
    force = np.array([10.0, 0.0, 0.0])
    moment = np.array([0.0, 0.0, 2.5])
    lin_vel = np.array([1.5, 0.0, 0.0])
    ang_vel = np.array([0.0, 0.0, -4.0])
    power = endpoint_interface_power_w(force, moment, lin_vel, ang_vel)
    # F·v = 15 W, M·ω = 2.5 * (-4) = -10 W
    assert power == pytest.approx(5.0, abs=1e-12)


def test_manufactured_planar_chain_robertson_winter_residual_near_zero(
    params, inertials
) -> None:
    """Rollout-derived planar chain closes RW interface power (same fixture as #8380)."""
    from scripts.research.proximal_distal_energy.run_experiments import (
        rollout_program,
        trace_accelerations,
    )
    from scripts.research.proximal_distal_energy.swing_model import (
        wrist_interface_powers,
    )
    from scripts.research.proximal_distal_energy.torque_programs import (
        restrain_then_drive_program,
    )

    t, q, v, u = rollout_program(
        params, restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    )
    qdd = trace_accelerations(params, q, v, u)
    powers = wrist_interface_powers(inertials, t, q, v, qdd, u)
    interface = powers["joint_force_power"] + powers["moment_power_on_club"]
    np.testing.assert_allclose(
        powers["club_energy_rate"],
        interface,
        rtol=0.0,
        atol=1e-9,
    )
    residuals = [
        segment_energy_balance_residual_w(
            mechanical_energy_rate_w=float(ke),
            interface_power_w=float(iface),
            gravity_power_w=float(grav),
            external_power_w=0.0,
        )
        for ke, iface, grav in zip(
            powers["club_kinetic_energy_rate"],
            interface,
            powers["gravity_power_on_club"],
            strict=True,
        )
    ]
    assert max(abs(r) for r in residuals) < 1e-6
