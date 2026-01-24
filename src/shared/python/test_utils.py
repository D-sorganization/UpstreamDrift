"""Test utilities for eliminating test code duplication.

This module provides reusable test patterns and fixtures.

Usage:
    from src.shared.python.test_utils import (
        skip_if_engine_unavailable,
        create_temp_model_file,
        assert_arrays_close,
    )

    @skip_if_engine_unavailable(EngineType.MUJOCO)
    def test_mujoco_feature():
        pass
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pytest

from src.shared.python.constants import GRAVITY_M_S2
from src.shared.python.engine_manager import EngineManager, EngineType
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def skip_if_engine_unavailable(engine_type: EngineType) -> Callable[[F], F]:
    """Decorator to skip test if engine is not available.

    Args:
        engine_type: Engine type to check

    Returns:
        Decorated test function

    Example:
        @skip_if_engine_unavailable(EngineType.MUJOCO)
        def test_mujoco_feature():
            pass
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = EngineManager()
            probe_result = manager.get_probe_result(engine_type)

            if not probe_result.is_available():
                pytest.skip(f"{engine_type.value} not installed")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def is_engine_available(engine_type: EngineType) -> bool:
    """Check if an engine is available.

    Args:
        engine_type: Engine type to check

    Returns:
        True if available, False otherwise

    Example:
        if is_engine_available(EngineType.DRAKE):
            # Use Drake
            pass
    """
    manager = EngineManager()
    probe_result = manager.get_probe_result(engine_type)
    return bool(probe_result.is_available())


def create_temp_model_file(content: str, suffix: str = ".xml") -> Path:
    """Create temporary model file for testing.

    Args:
        content: Model content
        suffix: File suffix (e.g., ".xml", ".urdf")

    Returns:
        Path to temporary file

    Example:
        model_path = create_temp_model_file(xml_content, ".xml")
        engine.load_from_path(str(model_path))
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return Path(f.name)


def assert_arrays_close(
    actual: np.ndarray,
    expected: np.ndarray,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = "",
) -> None:
    """Assert arrays are close with informative error message.

    Args:
        actual: Actual array
        expected: Expected array
        rtol: Relative tolerance
        atol: Absolute tolerance
        msg: Additional error message

    Raises:
        AssertionError: If arrays are not close

    Example:
        assert_arrays_close(result, expected, rtol=1e-3, msg="Position mismatch")
    """
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        error_msg = f"\n{msg}\n" if msg else "\n"
        error_msg += "Arrays not close:\n"
        error_msg += f"  Actual:   {actual}\n"
        error_msg += f"  Expected: {expected}\n"
        error_msg += f"  Max diff: {np.max(np.abs(actual - expected))}\n"
        error_msg += f"  Tolerance: rtol={rtol}, atol={atol}\n"
        error_msg += str(e)
        raise AssertionError(error_msg) from e


def assert_energy_conserved(
    energies: np.ndarray,
    initial_energy: float,
    tolerance: float = 0.01,
    msg: str = "",
) -> None:
    """Assert energy is conserved within tolerance.

    Args:
        energies: Array of energy values over time
        initial_energy: Initial energy value
        tolerance: Relative tolerance (e.g., 0.01 = 1%)
        msg: Additional error message

    Raises:
        AssertionError: If energy is not conserved

    Example:
        assert_energy_conserved(energies, initial_energy, tolerance=0.001)
    """
    max_deviation = np.max(np.abs(energies - initial_energy))
    percent_error = (max_deviation / abs(initial_energy)) * 100

    if percent_error > tolerance * 100:
        error_msg = f"\n{msg}\n" if msg else "\n"
        error_msg += "Energy not conserved:\n"
        error_msg += f"  Initial energy: {initial_energy:.6f}\n"
        error_msg += f"  Max deviation:  {max_deviation:.6f}\n"
        error_msg += f"  Percent error:  {percent_error:.4f}%\n"
        error_msg += f"  Tolerance:      {tolerance * 100:.4f}%\n"
        raise AssertionError(error_msg)


def assert_momentum_conserved(
    momentums: np.ndarray,
    initial_momentum: float | np.ndarray,
    tolerance: float = 1e-6,
    msg: str = "",
) -> None:
    """Assert momentum is conserved within tolerance.

    Args:
        momentums: Array of momentum values over time
        initial_momentum: Initial momentum value
        tolerance: Absolute tolerance
        msg: Additional error message

    Raises:
        AssertionError: If momentum is not conserved

    Example:
        assert_momentum_conserved(momentums, initial_momentum, tolerance=1e-6)
    """
    max_deviation = np.max(np.abs(momentums - initial_momentum))

    if max_deviation > tolerance:
        error_msg = f"\n{msg}\n" if msg else "\n"
        error_msg += "Momentum not conserved:\n"
        error_msg += f"  Initial momentum: {initial_momentum}\n"
        error_msg += f"  Max deviation:    {max_deviation:.6e}\n"
        error_msg += f"  Tolerance:        {tolerance:.6e}\n"
        raise AssertionError(error_msg)


class MockEngine:
    """Mock physics engine for testing.

    Example:
        engine = MockEngine()
        engine.load_from_path("model.xml")
        engine.step()
    """

    def __init__(self) -> None:
        """Initialize mock engine."""
        self.model_name_str = "MockModel"
        self.is_loaded = False
        self.step_count = 0
        self.time = 0.0

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.model_name_str

    def load_from_path(self, path: str) -> None:
        """Mock load from path."""
        self.model_name_str = Path(path).stem
        self.is_loaded = True
        logger.debug(f"Mock loaded: {path}")

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """Mock load from string."""
        self.model_name_str = "StringModel"
        self.is_loaded = True
        logger.debug("Mock loaded from string")

    def step(self, dt: float | None = None) -> None:
        """Mock simulation step."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        dt = dt or 0.001
        self.time += dt
        self.step_count += 1

    def reset(self) -> None:
        """Mock reset."""
        self.time = 0.0
        self.step_count = 0

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Mock get state."""
        return np.zeros(3), np.zeros(3)

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Mock set state."""
        pass


def create_simple_pendulum_xml(
    length: float = 1.0,
    mass: float = 1.0,
    gravity: float = GRAVITY_M_S2,  # DRY: Use centralized constant
) -> str:
    """Create simple pendulum model XML for testing.

    Args:
        length: Pendulum length [m]
        mass: Pendulum mass [kg]
        gravity: Gravity [m/s²]

    Returns:
        XML string

    Example:
        xml = create_simple_pendulum_xml(length=1.0, mass=1.0)
        model_path = create_temp_model_file(xml, ".xml")
    """
    return f"""
    <mujoco>
        <option timestep="0.001" gravity="0 0 -{gravity}" integrator="RK4"/>
        <worldbody>
            <body>
                <joint name="pin" type="hinge" axis="0 1 0" pos="0 0 0"/>
                <geom type="cylinder" fromto="0 0 0 0 0 -{length}" size="0.02" mass="0"/>
                <body pos="0 0 -{length}">
                    <geom type="sphere" size="0.1" mass="{mass}"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """


def create_simple_urdf(
    name: str = "simple_robot",
    mass: float = 1.0,
) -> str:
    """Create simple URDF for testing.

    Args:
        name: Robot name
        mass: Link mass [kg]

    Returns:
        URDF string

    Example:
        urdf = create_simple_urdf("test_robot")
        model_path = create_temp_model_file(urdf, ".urdf")
    """
    return f"""<?xml version="1.0"?>
    <robot name="{name}">
        <link name="base_link">
            <inertial>
                <mass value="{mass}"/>
                <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
            </inertial>
            <visual>
                <geometry>
                    <box size="0.1 0.1 0.1"/>
                </geometry>
            </visual>
        </link>
    </robot>
    """


class PerformanceTimer:
    """Context manager for timing code execution.

    Example:
        with PerformanceTimer("Model loading"):
            engine.load_from_path("model.xml")
    """

    def __init__(self, name: str, log_result: bool = True):
        """Initialize timer.

        Args:
            name: Timer name
            log_result: Whether to log the result
        """
        self.name = name
        self.log_result = log_result
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> PerformanceTimer:
        """Start timer."""
        import time

        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop timer and log result."""
        import time

        self.elapsed = time.perf_counter() - self.start_time

        if self.log_result:
            logger.info(f"{self.name}: {self.elapsed:.4f}s")


def parametrize_engines(
    engines: list[EngineType] | None = None,
) -> pytest.MarkDecorator:
    """Parametrize test across multiple engines.

    Args:
        engines: List of engines to test (default: all available)

    Returns:
        pytest.mark.parametrize decorator

    Example:
        @parametrize_engines([EngineType.MUJOCO, EngineType.DRAKE])
        def test_feature(engine_type):
            engine = create_engine(engine_type)
            # Test code
    """
    if engines is None:
        engines = [
            EngineType.MUJOCO,
            EngineType.DRAKE,
            EngineType.PINOCCHIO,
        ]

    # Filter to only available engines
    available_engines = [e for e in engines if is_engine_available(e)]

    return pytest.mark.parametrize("engine_type", available_engines)
