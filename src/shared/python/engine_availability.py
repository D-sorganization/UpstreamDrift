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

# Data & I/O library flags
NUMPY_AVAILABLE: bool = False
PANDAS_AVAILABLE: bool = False
PYARROW_AVAILABLE: bool = False
FASTPARQUET_AVAILABLE: bool = False
HDF5_AVAILABLE: bool = False
EZC3D_AVAILABLE: bool = False
YAML_AVAILABLE: bool = False

# GUI library flags
PYQT6_AVAILABLE: bool = False
PYQT5_AVAILABLE: bool = False
PYSIDE6_AVAILABLE: bool = False

# Additional optional library flags
PIL_AVAILABLE: bool = False
CV2_AVAILABLE: bool = False
STRUCTLOG_AVAILABLE: bool = False
GYMNASIUM_AVAILABLE: bool = False
GYM_AVAILABLE: bool = False
URDFPY_AVAILABLE: bool = False
TRIMESH_AVAILABLE: bool = False
MOVIEPY_AVAILABLE: bool = False
CX_FREEZE_AVAILABLE: bool = False
JSONSCHEMA_AVAILABLE: bool = False
COLORAMA_AVAILABLE: bool = False
TQDM_AVAILABLE: bool = False
REQUESTS_AVAILABLE: bool = False

# Scientific computing and ML library flags
NUMBA_AVAILABLE: bool = False
FASTDTW_AVAILABLE: bool = False
SKLEARN_AVAILABLE: bool = False
PYQTGRAPH_AVAILABLE: bool = False
SYMPY_AVAILABLE: bool = False
SKIMAGE_AVAILABLE: bool = False
SEABORN_AVAILABLE: bool = False

# Check NumPy (almost always available but good to check)
try:
    import numpy  # noqa: F401

    NUMPY_AVAILABLE = True
except ImportError:
    pass

# Check Pandas
try:
    import pandas  # noqa: F401

    PANDAS_AVAILABLE = True
except ImportError:
    pass

# Check PyArrow
try:
    import pyarrow  # noqa: F401

    PYARROW_AVAILABLE = True
except ImportError:
    pass

# Check fastparquet
try:
    import fastparquet  # noqa: F401

    FASTPARQUET_AVAILABLE = True
except ImportError:
    pass

# Check h5py (HDF5)
try:
    import h5py  # noqa: F401

    HDF5_AVAILABLE = True
except ImportError:
    pass

# Check ezc3d (C3D file format)
try:
    import ezc3d  # noqa: F401

    EZC3D_AVAILABLE = True
except ImportError:
    pass

# Check PyYAML
try:
    import yaml  # noqa: F401

    YAML_AVAILABLE = True
except ImportError:
    pass

# Check PyQt6
try:
    from PyQt6 import QtWidgets  # noqa: F401

    PYQT6_AVAILABLE = True
except (ImportError, OSError):
    pass

# Check PyQt5
try:
    from PyQt5 import QtWidgets  # noqa: F401

    PYQT5_AVAILABLE = True
except (ImportError, OSError):
    pass

# Check PySide6
try:
    from PySide6 import QtWidgets  # noqa: F401

    PYSIDE6_AVAILABLE = True
except (ImportError, OSError):
    pass

# Derived availability flags
PARQUET_AVAILABLE: bool = PYARROW_AVAILABLE or FASTPARQUET_AVAILABLE
QT_AVAILABLE: bool = PYQT6_AVAILABLE or PYQT5_AVAILABLE or PYSIDE6_AVAILABLE

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

# Check PIL/Pillow
try:
    from PIL import Image  # noqa: F401

    PIL_AVAILABLE = True
except ImportError:
    pass

# Check OpenCV
try:
    import cv2  # noqa: F401

    CV2_AVAILABLE = True
except ImportError:
    pass

# Check structlog
try:
    import structlog  # noqa: F401

    STRUCTLOG_AVAILABLE = True
except ImportError:
    pass

# Check gymnasium (new OpenAI Gym)
try:
    import gymnasium  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    pass

# Check gym (legacy OpenAI Gym)
try:
    import gym  # noqa: F401

    GYM_AVAILABLE = True
except ImportError:
    pass

# Check urdfpy
try:
    import urdfpy  # noqa: F401

    URDFPY_AVAILABLE = True
except ImportError:
    pass

# Check trimesh
try:
    import trimesh  # noqa: F401

    TRIMESH_AVAILABLE = True
except ImportError:
    pass

# Check moviepy
try:
    import moviepy  # noqa: F401

    MOVIEPY_AVAILABLE = True
except ImportError:
    pass

# Check cx_Freeze (for installers)
try:
    import cx_Freeze  # noqa: F401

    CX_FREEZE_AVAILABLE = True
except ImportError:
    pass

# Check jsonschema
try:
    import jsonschema  # noqa: F401

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    pass

# Check colorama
try:
    import colorama  # noqa: F401

    COLORAMA_AVAILABLE = True
except ImportError:
    pass

# Check tqdm
try:
    import tqdm  # noqa: F401

    TQDM_AVAILABLE = True
except ImportError:
    pass

# Check requests
try:
    import requests  # noqa: F401

    REQUESTS_AVAILABLE = True
except ImportError:
    pass

# Check Numba (JIT compilation)
try:
    import numba  # noqa: F401

    NUMBA_AVAILABLE = True
except ImportError:
    pass

# Check fastdtw (Dynamic Time Warping)
try:
    import fastdtw  # noqa: F401

    FASTDTW_AVAILABLE = True
except ImportError:
    pass

# Check scikit-learn
try:
    import sklearn  # noqa: F401

    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# Check pyqtgraph
try:
    import pyqtgraph  # noqa: F401

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    pass

# Check SymPy (symbolic math)
try:
    import sympy  # noqa: F401

    SYMPY_AVAILABLE = True
except ImportError:
    pass

# Check scikit-image
try:
    import skimage  # noqa: F401

    SKIMAGE_AVAILABLE = True
except ImportError:
    pass

# Check seaborn
try:
    import seaborn  # noqa: F401

    SEABORN_AVAILABLE = True
except ImportError:
    pass

# Derived availability flags
GYM_ANY_AVAILABLE: bool = GYMNASIUM_AVAILABLE or GYM_AVAILABLE

# Mapping of engine names to availability flags
_ENGINE_FLAGS: dict[str, bool] = {
    # Physics engines
    "mujoco": MUJOCO_AVAILABLE,
    "pinocchio": PINOCCHIO_AVAILABLE,
    "drake": DRAKE_AVAILABLE,
    "opensim": OPENSIM_AVAILABLE,
    "myosuite": MYOSUITE_AVAILABLE,
    "dm_control": DM_CONTROL_AVAILABLE,
    # ML frameworks
    "pytorch": PYTORCH_AVAILABLE,
    "torch": PYTORCH_AVAILABLE,  # Alias
    "tensorflow": TENSORFLOW_AVAILABLE,
    "tf": TENSORFLOW_AVAILABLE,  # Alias
    # Pose estimation
    "mediapipe": MEDIAPIPE_AVAILABLE,
    "myoconverter": MYOCONVERTER_AVAILABLE,
    "openpose": OPENPOSE_AVAILABLE,
    # Scientific computing
    "scipy": SCIPY_AVAILABLE,
    "matplotlib": MATPLOTLIB_AVAILABLE,
    "numpy": NUMPY_AVAILABLE,
    "pandas": PANDAS_AVAILABLE,
    # Data I/O
    "pyarrow": PYARROW_AVAILABLE,
    "fastparquet": FASTPARQUET_AVAILABLE,
    "parquet": PARQUET_AVAILABLE,
    "hdf5": HDF5_AVAILABLE,
    "h5py": HDF5_AVAILABLE,  # Alias
    "ezc3d": EZC3D_AVAILABLE,
    "c3d": EZC3D_AVAILABLE,  # Alias
    "yaml": YAML_AVAILABLE,
    "pyyaml": YAML_AVAILABLE,  # Alias
    # GUI frameworks
    "pyqt6": PYQT6_AVAILABLE,
    "pyqt5": PYQT5_AVAILABLE,
    "pyside6": PYSIDE6_AVAILABLE,
    "qt": QT_AVAILABLE,
    # Image/Video processing
    "pil": PIL_AVAILABLE,
    "pillow": PIL_AVAILABLE,  # Alias
    "cv2": CV2_AVAILABLE,
    "opencv": CV2_AVAILABLE,  # Alias
    "moviepy": MOVIEPY_AVAILABLE,
    # Robotics/Simulation
    "urdfpy": URDFPY_AVAILABLE,
    "trimesh": TRIMESH_AVAILABLE,
    "gymnasium": GYMNASIUM_AVAILABLE,
    "gym": GYM_AVAILABLE,
    "gym_any": GYM_ANY_AVAILABLE,
    # Utilities
    "structlog": STRUCTLOG_AVAILABLE,
    "cx_freeze": CX_FREEZE_AVAILABLE,
    "jsonschema": JSONSCHEMA_AVAILABLE,
    "colorama": COLORAMA_AVAILABLE,
    "tqdm": TQDM_AVAILABLE,
    "requests": REQUESTS_AVAILABLE,
    # Scientific computing and ML
    "numba": NUMBA_AVAILABLE,
    "fastdtw": FASTDTW_AVAILABLE,
    "sklearn": SKLEARN_AVAILABLE,
    "scikit-learn": SKLEARN_AVAILABLE,  # Alias
    "pyqtgraph": PYQTGRAPH_AVAILABLE,
    "sympy": SYMPY_AVAILABLE,
    "skimage": SKIMAGE_AVAILABLE,
    "scikit-image": SKIMAGE_AVAILABLE,  # Alias
    "seaborn": SEABORN_AVAILABLE,
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
