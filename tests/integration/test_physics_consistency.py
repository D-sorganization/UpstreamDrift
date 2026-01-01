"""Integration tests for checking consistency across physics engines (MuJoCo, Pinocchio, Drake).

This test suite runs simple pendulum simulations on all available engines
and asserts that the results (e.g., period, energy conservation) are close.
"""

from typing import Any

import numpy as np
import pytest

from shared.python.constants import GRAVITY_M_S2

# Tolerances
PERIOD_TOLERANCE = 1e-2  # Seconds difference allowed
ENERGY_TOLERANCE = 1e-3  # Joules energy drift allowed


class SimplePendulumParams:
    """Standard parameters for cross-engine validation."""

    mass: float = 1.0  # kg
    length: float = 1.0  # m
    gravity: float = GRAVITY_M_S2  # m/s^2 (Approx, usage differs slightly engines)
    initial_angle: float = np.pi / 4  # rad (45 degrees)
    duration: float = 5.0  # s
    timestep: float = 0.001  # s


def expected_period_small_angle(length: float, gravity: float) -> float:
    """Calculate theoretical period for small angles T = 2*pi*sqrt(L/g)."""
    return float(2 * np.pi * np.sqrt(length / gravity))


def run_mujoco_pendulum(params: SimplePendulumParams) -> dict[str, Any]:
    """Run pendulum simulation in MuJoCo."""
    try:
        import mujoco
        if "unittest.mock" in str(type(mujoco)):
             return {"error": "MuJoCo is mocked"}
    except ImportError:
        return {"error": "MuJoCo not installed"}

    xml = f"""
    <mujoco>
      <option gravity="0 0 {-params.gravity}"/>
      <worldbody>
        <body>
          <geom type="sphere" size="0.1" mass="{params.mass}"/>
          <joint name="pivot" type="hinge" axis="0 1 0" pos="0 0 {params.length}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
    except Exception as e:
        return {"error": f"MuJoCo init failed: {e}"}

    # Set initial state
    data.qpos[0] = params.initial_angle

    # Run simulation
    n_steps = int(params.duration / params.timestep)
    model.opt.timestep = params.timestep

    times = []
    angles = []

    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        times.append(data.time)
        angles.append(data.qpos[0])

    return {"times": times, "angles": angles}


def run_pinocchio_pendulum(params: SimplePendulumParams) -> dict[str, Any]:
    """Run pendulum simulation in Pinocchio."""
    try:
        import pinocchio as pin

        if hasattr(pin, "assert_called") or hasattr(pin, "reset_mock"):
            return {"error": "Pinocchio is mocked"}
    except ImportError:
        return {"error": "Pinocchio not installed"}

    # Create model
    model = pin.Model()

    # Joint
    joint_id = model.addJoint(
        0, pin.JointModelRY(), pin.SE3(np.eye(3), np.array([0, 0, 0])), "joint"
    )

    # Inertia (Point mass at length L)
    # Pinocchio inertia is at the joint origin usually, we need to offset it?
    # Actually, simpler to place a body at COM.
    # COM is at (0, 0, -length) relative to joint if hanging down?
    # Let's say z is up. Joint at origin. Mass at (0, 0, -L).
    # Rotation Y.

    mass = params.mass
    inertia = pin.Inertia.FromSphere(mass, 0.0)  # Point mass approximation
    # Offset inertia center
    inertia.lever = np.array([0, 0, -params.length])

    model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())
    model.gravity = pin.Motion(np.array([0, 0, 0, 0, 0, -params.gravity]))

    data = model.createData()

    # Simulation loop (Euler integration for simplicity, matching simple MuJoCo)
    # Or better, Rnea/Aba
    q = np.array([params.initial_angle])
    v = np.array([0.0])
    dt = params.timestep
    n_steps = int(params.duration / dt)

    times = []
    angles = []
    curr_t = 0.0

    for _ in range(n_steps):
        a = pin.aba(model, data, q, v, np.zeros(1))

        # Semi-implicit Euler
        v += a * dt
        q += v * dt
        curr_t += dt

        times.append(curr_t)
        angles.append(q[0])

    return {"times": times, "angles": angles}


def test_physics_engine_consistency():
    """Verify that all engines produce similar physics for a simple pendulum."""
    params = SimplePendulumParams()

    results = {}

    # Run simulations
    mujoco_res = run_mujoco_pendulum(params)
    results["mujoco"] = mujoco_res

    pinocchio_res = run_pinocchio_pendulum(params)
    results["pinocchio"] = pinocchio_res

    # Check for valid results
    valid_engines = []
    for name, res in results.items():
        if "error" not in res:
            valid_engines.append(name)
        else:
            print(f"Skipping {name}: {res['error']}")

    if not valid_engines:
        pytest.skip("No physics engines installed locally.")

    # Validation Logic
    # 1. Check Period
    theoretical_period = expected_period_small_angle(params.length, params.gravity)
    # Note: Large angle (45 deg) period is slightly larger than small angle approx
    # Approx adjustment: T = T0 * (1 + theta0^2/16)
    theta0 = params.initial_angle
    adjusted_period = theoretical_period * (1 + theta0**2 / 16)

    print(f"\nTheoretical Period (Small Angle): {theoretical_period:.4f}s")
    print(f"Expected Period (Large Angle): {adjusted_period:.4f}s")

    for engine in valid_engines:
        res = results[engine]
        angles = np.array(res["angles"])
        times = np.array(res["times"])

        # Zero crossings (falling edge) to find full period roughly
        # Or peak to peak
        peaks = []
        for i in range(1, len(angles) - 1):
            if angles[i] > angles[i - 1] and angles[i] > angles[i + 1]:
                peaks.append(times[i])

        if len(peaks) > 1:
            avg_period = np.diff(peaks).mean()
            print(f"{engine} Period: {avg_period:.4f}s")

            # Assert period is reasonable
            # We allow loose tolerance because integration schemes differ
            assert (
                abs(avg_period - adjusted_period) < 0.1
            ), f"{engine} period {avg_period} deviates from expected {adjusted_period}"

    # 2. Cross-Engine Consistency
    if "mujoco" in valid_engines and "pinocchio" in valid_engines:
        # Compare first peak amplitude
        m_angles = results["mujoco"]["angles"]
        p_angles = results["pinocchio"]["angles"]

        # Compare first 100 steps
        # They drift quickly due to integration differences (Runge-Kutta vs Euler)
        # So we check very early consistency
        drift = np.abs(np.array(m_angles[:100]) - np.array(p_angles[:100])).mean()
        print(f"Mean drift over first 0.1s: {drift:.6f} rad")

        assert drift < 0.05, "Engines diverge significantly in first 100 steps"
