#!/usr/bin/env python3
"""End-to-end example: load a physics engine and run a short simulation.

This example uses :class:`MockPhysicsEngine`, a dependency-light
(``numpy``-only) engine that implements the same protocol as the real
backends (MuJoCo, Drake, Pinocchio, ...). It is the recommended starting
point because it runs anywhere — no engine binaries or model files required.

Run it directly from the repository root::

    python docs/examples/run_mock_engine_sim.py

Swap ``get_mock_engine()`` for a real loader (see
``docs/engines/engine_selection_guide.md``) to drive an actual backend; the
load -> set-control -> step -> read-state loop below is identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make ``src`` importable when the script is run directly from a checkout.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.engine_core.mock_engine import (  # noqa: E402
    get_mock_engine,
)


def run_simulation(num_steps: int = 100, dt: float = 0.01) -> np.ndarray:
    """Load the mock engine and integrate a constant-torque swing.

    Args:
        num_steps: Number of integration steps to advance.
        dt: Timestep in seconds.

    Returns:
        The recorded joint-position trajectory with shape
        ``(num_steps + 1, num_joints)``.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    engine = get_mock_engine()
    engine.load_model("mock_golfer.urdf")
    engine.reset()

    num_joints = len(engine.get_joint_names())
    # Apply a small constant torque to every joint.
    engine.set_control(np.full(num_joints, 0.5))

    positions, _ = engine.get_state()
    trajectory = [positions]
    for _ in range(num_steps):
        engine.step(dt)
        positions, _ = engine.get_state()
        trajectory.append(positions)

    return np.asarray(trajectory)


def main() -> int:
    """Run the simulation and print a short summary."""
    trajectory = run_simulation()
    final = trajectory[-1]
    print(
        f"Simulated {trajectory.shape[0] - 1} steps over {trajectory.shape[1]} joints."
    )
    print(f"Final joint positions: {np.round(final, 4)}")
    # The constant torque should drive every joint away from zero.
    assert np.all(np.abs(final) > 0.0), "expected non-trivial motion"
    print("Mock engine simulation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
