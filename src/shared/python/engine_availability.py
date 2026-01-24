"""Engine availability checking utilities.

This module consolidates the common pattern of checking for optional
physics engine imports across the codebase, addressing DRY violations
identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.engine_availability import (
        MUJOCO_AVAILABLE,
        PINOCCHIO_AVAILABLE,
        is_engine_available,
        require_engine,
    )

    # Simple boolean check
    if MUJOCO_AVAILABLE:
        import mujoco
        ...

    # Function-based check
    if is_engine_available("drake"):
        ...

    # Decorator for tests
    @require_engine("pinocchio")
    def test_pinocchio_jacobian():
        ...
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

# Engine availability flags - evaluated at import time
MUJOCO_AVAILABLE: bool = False
PINOCCHIO_AVAILABLE: bool = False
DRAKE_AVAILABLE: bool = False
OPENSIM_AVAILABLE: bool = False
MYOSUITE_AVAILABLE: bool = False
MEDIAPIPE_AVAILABLE: bool = False
DM_CONTROL_AVAILABLE: bool = False
PYTORCH_AVAILABLE: bool = False
TENSORFLOW_AVAILABLE: bool = False
MYOCONVERTER_AVAILABLE: bool = False
OPENPOSE_AVAILABLE: bool = False
SCIPY_AVAILABLE: bool = False
MATPLOTLIB_AVAILABLE: bool = False

# Check MuJoCo
try:
    import mujoco  # noqa: F401

    MUJOCO_AVAILABLE = True
except ImportError:
    pass

# Check Pinocchio
try:
    import pinocchio  # noqa: F401

    PINOCCHIO_AVAILABLE = True
except ImportError:
    pass

# Check Drake
try:
    import pydrake  # noqa: F401

    DRAKE_AVAILABLE = True
except ImportError:
    pass

# Check OpenSim
try:
    import opensim  # noqa: F401

    OPENSIM_AVAILABLE = True
except ImportError:
    pass

# Check MyoSuite
try:
    import myosuite  # noqa: F401

    MYOSUITE_AVAILABLE = True
except ImportError:
    pass

# Check MediaPipe
try:
    import mediapipe  # noqa: F401

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

# Check dm_control
try:
    import dm_control  # noqa: F401

    DM_CONTROL_AVAILABLE = True
except ImportError:
    pass

# Check PyTorch
try:
    import torch  # noqa: F401

    PYTORCH_AVAILABLE = True
except ImportError:
    pass

# Check TensorFlow
try:
    import tensorflow  # noqa: F401

    TENSORFLOW_AVAILABLE = True
except ImportError:
    pass

# Check MyoConverter
try:
    import myoconverter  # noqa: F401

    MYOCONVERTER_AVAILABLE = True
except ImportError:
    pass

# Check OpenPose
try:
    import openpose  # noqa: F401

    OPENPOSE_AVAILABLE = True
except ImportError:
    pass

# Check SciPy
try:
    import scipy  # noqa: F401

    SCIPY_AVAILABLE = True
except ImportError:
    pass

# Check Matplotlib
try:
    import matplotlib  # noqa: F401

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass

# Check PyArrow (for parquet support)
PYARROW_AVAILABLE: bool = False
try:
    import pyarrow  # noqa: F401

    PYARROW_AVAILABLE = True
except ImportError:
    pass

# Check FastParquet (alternative parquet support)
FASTPARQUET_AVAILABLE: bool = False
try:
    import fastparquet  # noqa: F401

    FASTPARQUET_AVAILABLE = True
except ImportError:
    pass

# Check tables/pytables (for HDF5 support)
HDF5_AVAILABLE: bool = False
try:
    import tables  # noqa: F401

    HDF5_AVAILABLE = True
except ImportError:
    pass

# Convenience flags
PARQUET_AVAILABLE: bool = PYARROW_AVAILABLE or FASTPARQUET_AVAILABLE

# Mapping of engine names to availability flags
_ENGINE_FLAGS: dict[str, bool] = {
    "mujoco": MUJOCO_AVAILABLE,
    "pinocchio": PINOCCHIO_AVAILABLE,
    "drake": DRAKE_AVAILABLE,
    "opensim": OPENSIM_AVAILABLE,
    "myosuite": MYOSUITE_AVAILABLE,
    "mediapipe": MEDIAPIPE_AVAILABLE,
    "dm_control": DM_CONTROL_AVAILABLE,
    "pytorch": PYTORCH_AVAILABLE,
    "torch": PYTORCH_AVAILABLE,  # Alias
    "tensorflow": TENSORFLOW_AVAILABLE,
    "tf": TENSORFLOW_AVAILABLE,  # Alias
    "myoconverter": MYOCONVERTER_AVAILABLE,
    "openpose": OPENPOSE_AVAILABLE,
    "scipy": SCIPY_AVAILABLE,
    "matplotlib": MATPLOTLIB_AVAILABLE,
    "pyarrow": PYARROW_AVAILABLE,
    "fastparquet": FASTPARQUET_AVAILABLE,
    "parquet": PARQUET_AVAILABLE,
    "hdf5": HDF5_AVAILABLE,
    "tables": HDF5_AVAILABLE,  # Alias
}


def is_engine_available(engine_name: str) -> bool:
    """Check if a physics engine or library is available.

    Args:
        engine_name: Name of the engine (case-insensitive).

    Returns:
        True if the engine is importable, False otherwise.

    Example:
        if is_engine_available("mujoco"):
            import mujoco
            model = mujoco.MjModel.from_xml_path(...)
    """
    return _ENGINE_FLAGS.get(engine_name.lower(), False)


def get_available_engines() -> list[str]:
    """Get list of all available physics engines.

    Returns:
        List of engine names that are importable.
    """
    return [name for name, available in _ENGINE_FLAGS.items() if available]


def get_unavailable_engines() -> list[str]:
    """Get list of unavailable physics engines.

    Returns:
        List of engine names that are not importable.
    """
    return [name for name, available in _ENGINE_FLAGS.items() if not available]


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def require_engine(engine_name: str, reason: str | None = None) -> Callable[[F], F]:
    """Decorator to skip test/function if engine is not available.

    This is primarily intended for use with pytest tests but can be used
    with any callable.

    Args:
        engine_name: Name of the required engine.
        reason: Optional reason message for skipping.

    Returns:
        Decorated function that skips if engine unavailable.

    Example:
        @require_engine("pinocchio")
        def test_pinocchio_forward_kinematics():
            import pinocchio
            ...

        @require_engine("mujoco", reason="MuJoCo required for physics sim")
        def run_simulation():
            ...
    """
    skip_reason = reason or f"{engine_name} not installed"

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_engine_available(engine_name):
                # Try to use pytest.skip if available (for test functions)
                try:
                    import pytest

                    pytest.skip(skip_reason)
                except ImportError:
                    # If not in pytest, raise an ImportError
                    raise ImportError(
                        f"Required engine '{engine_name}' is not available. "
                        f"{skip_reason}"
                    ) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def skip_if_unavailable(engine_name: str) -> Any:
    """Create a pytest skip marker for unavailable engines.

    This is a convenience function for creating pytest markers.

    Args:
        engine_name: Name of the engine to check.

    Returns:
        pytest.mark.skipif marker or raises ImportError if pytest unavailable.

    Example:
        import pytest
        from src.shared.python.engine_availability import skip_if_unavailable

        @skip_if_unavailable("mujoco")
        def test_mujoco_jacobian():
            ...
    """
    try:
        import pytest

        return pytest.mark.skipif(
            not is_engine_available(engine_name),
            reason=f"{engine_name} not installed",
        )
    except ImportError:
        raise ImportError(
            "pytest is required for skip_if_unavailable. "
            "Use require_engine decorator instead."
        ) from None
