"""Shared fixtures for cross-engine validation tests.

This module provides pytest fixtures for loading test models into
different physics engines and validating cross-engine consistency.

Fixtures follow Guideline M2 (deterministic seeds, gold-standard models)
and P3 (tolerance-based validation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.constants import GRAVITY_M_S2
from src.shared.python.engine_availability import (
    DRAKE_AVAILABLE,
    MUJOCO_AVAILABLE,
    PINOCCHIO_AVAILABLE,
)
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# Fixture model paths
FIXTURES_DIR = Path(__file__).parent / "models"
SIMPLE_PENDULUM_URDF = FIXTURES_DIR / "simple_pendulum.urdf"
DOUBLE_PENDULUM_URDF = FIXTURES_DIR / "double_pendulum.urdf"

# Physical constants for analytical validation
# Source: NIST CODATA 2018

# Simple pendulum parameters (must match URDF)
PENDULUM_LENGTH_M = 1.0  # [m] Rod length
PENDULUM_MASS_KG = 1.0  # [kg] Point mass

# Analytical solution: ω = sqrt(g/L)
PENDULUM_NATURAL_FREQ_RAD_S = np.sqrt(GRAVITY_M_S2 / PENDULUM_LENGTH_M)

# Tolerances from Guideline P3
TOLERANCE_POSITION_M = 1e-6
TOLERANCE_VELOCITY_M_S = 1e-5
TOLERANCE_ACCELERATION_M_S2 = 1e-4
TOLERANCE_TORQUE_NM = 1e-3
TOLERANCE_JACOBIAN = 1e-8

# Indexed acceleration closure tolerance (Guideline M2)
TOLERANCE_CLOSURE_RAD_S2 = 1e-6


@dataclass
class EngineInstance:
    """Container for a loaded physics engine instance.

    Attributes:
        name: Engine identifier (e.g., "MuJoCo", "Drake", "Pinocchio")
        engine: Physics engine instance implementing PhysicsEngine protocol
        available: Whether this engine is available in the current environment
    """

    name: str
    engine: Any  # PhysicsEngine, but using Any for optional dependency handling
    available: bool


# DRY: Use centralized availability flags from engine_availability module
def _check_mujoco_available() -> bool:
    """Check if MuJoCo is available (delegates to engine_availability)."""
    return MUJOCO_AVAILABLE


def _check_drake_available() -> bool:
    """Check if Drake is available (delegates to engine_availability)."""
    return DRAKE_AVAILABLE


def _check_pinocchio_available() -> bool:
    """Check if Pinocchio is available (delegates to engine_availability)."""
    return PINOCCHIO_AVAILABLE


@pytest.fixture
def available_engines() -> dict[str, bool]:
    """Return dictionary of engine availability.

    Returns:
        Dictionary mapping engine names to availability status.

    Example:
        >>> engines = available_engines()
        >>> if engines["MuJoCo"]:
        ...     # MuJoCo tests can run
    """
    return {
        "MuJoCo": _check_mujoco_available(),
        "Drake": _check_drake_available(),
        "Pinocchio": _check_pinocchio_available(),
    }


@pytest.fixture
def simple_pendulum_path() -> Path:
    """Return path to simple pendulum URDF fixture."""
    if not SIMPLE_PENDULUM_URDF.exists():
        pytest.skip(f"Simple pendulum fixture not found: {SIMPLE_PENDULUM_URDF}")
    return SIMPLE_PENDULUM_URDF


@pytest.fixture
def double_pendulum_path() -> Path:
    """Return path to double pendulum URDF fixture."""
    if not DOUBLE_PENDULUM_URDF.exists():
        pytest.skip(f"Double pendulum fixture not found: {DOUBLE_PENDULUM_URDF}")
    return DOUBLE_PENDULUM_URDF


@pytest.fixture
def mujoco_pendulum(simple_pendulum_path: Path) -> EngineInstance:
    """Load simple pendulum into MuJoCo engine.

    Returns:
        EngineInstance with loaded MuJoCo engine or unavailable marker.
    """
    if not _check_mujoco_available():
        return EngineInstance(name="MuJoCo", engine=None, available=False)

    try:
        from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        engine = MuJoCoPhysicsEngine()
        engine.load_from_path(str(simple_pendulum_path))
        engine.reset()
        return EngineInstance(name="MuJoCo", engine=engine, available=True)
    except Exception as e:
        logger.warning(f"Failed to load MuJoCo pendulum: {e}")
        return EngineInstance(name="MuJoCo", engine=None, available=False)


@pytest.fixture
def drake_pendulum(simple_pendulum_path: Path) -> EngineInstance:
    """Load simple pendulum into Drake engine.

    Returns:
        EngineInstance with loaded Drake engine or unavailable marker.
    """
    if not _check_drake_available():
        return EngineInstance(name="Drake", engine=None, available=False)

    try:
        from src.engines.physics_engines.drake.python.drake_physics_engine import (
            DrakePhysicsEngine,
        )

        engine = DrakePhysicsEngine()
        engine.load_from_path(str(simple_pendulum_path))
        engine.reset()
        return EngineInstance(name="Drake", engine=engine, available=True)
    except Exception as e:
        logger.warning(f"Failed to load Drake pendulum: {e}")
        return EngineInstance(name="Drake", engine=None, available=False)


@pytest.fixture
def pinocchio_pendulum(simple_pendulum_path: Path) -> EngineInstance:
    """Load simple pendulum into Pinocchio engine.

    Returns:
        EngineInstance with loaded Pinocchio engine or unavailable marker.
    """
    if not _check_pinocchio_available():
        return EngineInstance(name="Pinocchio", engine=None, available=False)

    try:
        from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
            PinocchioPhysicsEngine,
        )

        engine = PinocchioPhysicsEngine()
        engine.load_from_path(str(simple_pendulum_path))
        engine.reset()
        return EngineInstance(name="Pinocchio", engine=engine, available=True)
    except Exception as e:
        logger.warning(f"Failed to load Pinocchio pendulum: {e}")
        return EngineInstance(name="Pinocchio", engine=None, available=False)


@pytest.fixture
def all_available_pendulum_engines(
    mujoco_pendulum: EngineInstance,
    drake_pendulum: EngineInstance,
    pinocchio_pendulum: EngineInstance,
) -> list[EngineInstance]:
    """Return list of all available engine instances with pendulum loaded.

    Returns:
        List of EngineInstance objects that are available and loaded.
    """
    engines = [mujoco_pendulum, drake_pendulum, pinocchio_pendulum]
    available = [e for e in engines if e.available]

    if len(available) < 2:
        pytest.skip(
            f"Need at least 2 engines for cross-validation, got {len(available)}"
        )

    return available


def set_identical_state(
    engines: list[EngineInstance],
    q: np.ndarray,
    v: np.ndarray,
) -> None:
    """Set identical state across all engines.

    Args:
        engines: List of engine instances to synchronize.
        q: Joint positions to set.
        v: Joint velocities to set.
    """
    for eng in engines:
        if eng.available and eng.engine is not None:
            eng.engine.set_state(q, v)
            eng.engine.forward()


def get_states(
    engines: list[EngineInstance],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Get current states from all engines.

    Args:
        engines: List of engine instances.

    Returns:
        Dictionary mapping engine names to (q, v) tuples.
    """
    states = {}
    for eng in engines:
        if eng.available and eng.engine is not None:
            q, v = eng.engine.get_state()
            states[eng.name] = (q, v)
    return states


def compute_accelerations(engines: list[EngineInstance]) -> dict[str, np.ndarray]:
    """Compute forward dynamics accelerations from all engines.

    Args:
        engines: List of engine instances.

    Returns:
        Dictionary mapping engine names to acceleration arrays.

    Note:
        This computes drift acceleration (ZTCF) with zero control input.
        The equations of motion are: M(q) * qacc = tau - bias(q, v)
        For tau = 0: qacc = -M(q)^{-1} * bias(q, v)
        The negative sign follows from the bias-force convention used by
        MuJoCo, Drake, and Pinocchio.
    """
    accelerations = {}
    for eng in engines:
        if eng.available and eng.engine is not None:
            # Compute forward kinematics/dynamics
            eng.engine.forward()
            # Get acceleration from bias forces and mass matrix
            M = eng.engine.compute_mass_matrix()
            bias = eng.engine.compute_bias_forces()
            if M.size > 0 and bias.size > 0:
                # NOTE: The leading minus sign follows the bias-force convention
                # For tau = 0: M * qacc = -bias => qacc = -M^-1 * bias
                qacc = -np.linalg.solve(M, bias)
                accelerations[eng.name] = qacc
    return accelerations


def skip_if_insufficient_engines(
    engines: list[EngineInstance], min_count: int = 2
) -> None:
    """Skip test if insufficient engines are available.

    Args:
        engines: List of engine instances to check.
        min_count: Minimum required engines (default 2 for cross-validation).

    Raises:
        pytest.skip: If fewer than min_count engines are available.
    """
    available = [e for e in engines if e.available]
    if len(available) < min_count:
        pytest.skip(
            f"Need at least {min_count} engines for cross-validation, "
            f"got {len(available)}: {[e.name for e in available]}"
        )
