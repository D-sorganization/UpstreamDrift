"""Engine availability checking utilities.

This module consolidates the common pattern of checking for optional
physics engine imports across the codebase, addressing DRY violations
identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.engine_core.engine_availability import (
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
import importlib
import importlib.util
import logging
import os
import sys
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)


class EngineStatus(Enum):
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    BROKEN = "broken"


_engine_status_cache: dict[str, EngineStatus] = {}
_engine_error_cache: dict[str, Exception] = {}
_dependency_presence_cache: dict[str, bool] = {}

# Probe names whose installed distribution exposes a different top-level
# package than the name callers use (mirrors the special cases in
# ``_probe_engine`` without importing anything).
_PRESENCE_SPEC_OVERRIDES: dict[str, str] = {
    "drake": "pydrake",
    "tf": "tensorflow",
    "pillow": "PIL",
    "pyqt6": "PyQt6",
    "pyqt5": "PyQt5",
    "pyside6": "PySide6",
    "openpose": "pyopenpose",
}


def is_dependency_present(import_name: str) -> bool:
    """Cheap, metadata-only availability probe for a runtime dependency.

    Uses :func:`importlib.util.find_spec` on the *top-level* package so that
    no module code is executed. This is the probe the discovery/tile phase
    must use (#8934): importing ``pydrake.all`` or ``jaxsim`` for real costs
    seconds and hundreds of MB, and belongs only on the actual launch path
    (see :func:`get_engine_status` / ``_probe_engine`` for the deep probe).

    Args:
        import_name: Import name as used by the deep probe (may be dotted,
            e.g. ``"jaxsim.api"``; only the top-level package is checked so
            the parent package is never imported).

    Returns:
        True when the package's import metadata is present.

    Postcondition:
        Never imports ``import_name`` (nor its top-level package): if the
        module was absent from ``sys.modules`` before the call it remains
        absent after it.
    """
    if not import_name:
        raise ValueError("import_name must be a non-empty string")
    spec_name = _PRESENCE_SPEC_OVERRIDES.get(import_name, import_name)
    spec_name = spec_name.partition(".")[0]
    if spec_name in _dependency_presence_cache:
        return _dependency_presence_cache[spec_name]
    try:
        present = importlib.util.find_spec(spec_name) is not None
    except (ImportError, ValueError):
        # ValueError: a module in sys.modules with __spec__ = None
        # (e.g. an injected test stub). Treat as not reliably present.
        present = False
    _dependency_presence_cache[spec_name] = present
    return present


def _import_probe_target(import_name: str) -> None:  # noqa: C901
    """Import the probe target for an engine, raising ImportError when unusable.

    Split out of :func:`_probe_engine` (function-lines budget): the per-engine
    import dispatch, including the mocked-module guards that stop a
    unit-test MagicMock injected into ``sys.modules`` from being cached as a
    real, available engine.
    """
    if import_name == "drake":
        pydrake_module = importlib.import_module("pydrake")
        drake_all_module = importlib.import_module("pydrake.all")
        if (
            type(pydrake_module).__module__ == "unittest.mock"
            or type(drake_all_module).__module__ == "unittest.mock"
        ):
            raise ImportError("mocked pydrake module detected")
    elif import_name == "torch":
        importlib.import_module("torch")
    elif import_name == "tf":
        importlib.import_module("tensorflow")
    elif import_name == "pyopenpose":
        importlib.import_module("pyopenpose")  # import pyopenpose (dynamic)
    elif import_name == "h5py":
        importlib.import_module("h5py")
    elif import_name == "yaml":
        importlib.import_module("yaml")
    elif import_name == "pillow":
        importlib.import_module("PIL.Image")
    elif import_name == "pyqt6":
        importlib.import_module("PyQt6.QtWidgets")
    elif import_name == "pyqt5":
        importlib.import_module("PyQt5.QtWidgets")
    elif import_name == "pyside6":
        if sys.platform == "win32":
            raise ImportError(
                "pyside6 probe disabled on Windows to prevent DLL crashes"
            )
        importlib.import_module("PySide6.QtWidgets")
    elif import_name == "pinocchio":
        pin = importlib.import_module("pinocchio")
        if type(pin).__module__ == "unittest.mock":
            # A unit-test conftest may inject a MagicMock into
            # sys.modules["pinocchio"] so that import-only tests collect
            # without the heavy C-extension. MagicMock answers hasattr()
            # truthy for everything, so without this guard the probe would
            # cache the engine as AVAILABLE and poison every later test
            # that genuinely requires the real bindings.
            raise ImportError("mocked pinocchio module detected")
        if not hasattr(pin, "buildModelFromUrdf"):
            raise ImportError(
                "Incorrect pinocchio package (likely nose plugin). Please install pinocchio from conda-forge."
            )
    elif import_name == "mediapipe":
        # MediaPipeEstimator drives the Tasks API (mediapipe>=0.10, #9592).
        # Distinguish a legacy-only install from a missing one so the reason
        # is actionable; the model file is a separate runtime check that
        # load_model() reports through ModelError.
        importlib.import_module("mediapipe")
        try:
            importlib.import_module("mediapipe.tasks.python.vision")
        except ImportError as exc:
            raise ImportError(
                "mediapipe is installed but lacks the Tasks API "
                "(mediapipe.tasks.python.vision); upgrade to mediapipe>=0.10. "
                "MediaPipeEstimator no longer uses the legacy mp.solutions API."
            ) from exc
    else:
        module = importlib.import_module(import_name)
        if type(module).__module__ == "unittest.mock":
            # See the pinocchio branch above: a mocked module injected for
            # collection must not be reported as a real, available engine.
            raise ImportError(f"mocked {import_name} module detected")


def _probe_engine(
    engine_name: str,
    import_name: str | None = None,
    is_broken_check: Callable[[Exception], bool] | None = None,
) -> EngineStatus:
    name = engine_name.lower()
    if name in _engine_status_cache:
        return _engine_status_cache[name]

    if import_name is None:
        import_name = name

    # Windows specfic skip for MuJoCo
    if name == "mujoco":
        if (
            sys.platform == "win32"
            and sys.version_info >= (3, 13)
            and os.environ.get("SKIP_MUJOCO_IMPORT", "").lower() == "true"
        ):
            logger.warning(
                "Skipping MuJoCo import on Windows Python 3.13+ due to DLL issues. Set FORCE_MUJOCO_IMPORT=true to override."
            )
            _engine_status_cache[name] = EngineStatus.BROKEN
            return EngineStatus.BROKEN
        if "MUJOCO_PLUGIN_PATH" not in os.environ:
            os.environ["MUJOCO_PLUGIN_PATH"] = ""

    try:
        _import_probe_target(import_name)

        _engine_status_cache[name] = EngineStatus.AVAILABLE
    except ImportError as e:
        _engine_status_cache[name] = EngineStatus.NOT_INSTALLED
        _engine_error_cache[name] = e
    except (OSError, RuntimeError, AttributeError, TypeError) as e:
        # Non-ImportError failures during engine load (e.g. missing shared libs,
        # DLL errors on Windows, broken C extensions, or bad package state).
        if is_broken_check is None or is_broken_check(e):
            _engine_status_cache[name] = EngineStatus.BROKEN
            _engine_error_cache[name] = e
            logger.warning("%s loading failed: %s", import_name, e)
        else:
            _engine_status_cache[name] = EngineStatus.NOT_INSTALLED
            _engine_error_cache[name] = e

    return _engine_status_cache[name]


_MODULE_MAPPING = {
    "mujoco": "mujoco",
    "pinocchio": "pinocchio",
    "jaxsim": "jaxsim",
    "drake": "drake",
    "opensim": "opensim",
    "myosuite": "myosuite",
    "dm_control": "dm_control",
    "pytorch": "torch",
    "torch": "torch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "mediapipe": "mediapipe",
    "myoconverter": "myoconverter",
    "openpose": "pyopenpose",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "fastparquet": "fastparquet",
    "hdf5": "h5py",
    "h5py": "h5py",
    "ezc3d": "ezc3d",
    "c3d": "ezc3d",
    "c3d_pkg": "c3d",
    "yaml": "yaml",
    "pyyaml": "yaml",
    "pyqt6": "pyqt6",
    "pyqt5": "pyqt5",
    "pyside6": "pyside6",
    "pytest_qt": "pytestqt",
    "pytestqt": "pytestqt",
    "pil": "pillow",
    "pillow": "pillow",
    "cv2": "cv2",
    "opencv": "cv2",
    "moviepy": "moviepy",
    "urdfpy": "urdfpy",
    "trimesh": "trimesh",
    "gymnasium": "gymnasium",
    "gym": "gym",
    "structlog": "structlog",
    "cx_freeze": "cx_Freeze",
    "jsonschema": "jsonschema",
    "colorama": "colorama",
    "tqdm": "tqdm",
    "requests": "requests",
    "numba": "numba",
    "fastdtw": "fastdtw",
    "sklearn": "sklearn",
    "scikit-learn": "sklearn",
    "pyqtgraph": "pyqtgraph",
    "sympy": "sympy",
    "skimage": "skimage",
    "scikit-image": "skimage",
    "seaborn": "seaborn",
}

_ENGINE_FLAGS: dict[str, bool] = dict.fromkeys(
    _MODULE_MAPPING, False
)  # availability populated lazily via get_engine_status at call time


def get_engine_status(engine_name: str) -> EngineStatus:
    name = engine_name.lower()

    if name == "parquet":
        return (
            EngineStatus.AVAILABLE
            if (
                get_engine_status("pyarrow") == EngineStatus.AVAILABLE
                or get_engine_status("fastparquet") == EngineStatus.AVAILABLE
            )
            else EngineStatus.NOT_INSTALLED
        )
    if name == "c3d_any":
        return (
            EngineStatus.AVAILABLE
            if (
                get_engine_status("ezc3d") == EngineStatus.AVAILABLE
                or get_engine_status("c3d_pkg") == EngineStatus.AVAILABLE
            )
            else EngineStatus.NOT_INSTALLED
        )
    if name == "qt":
        return (
            EngineStatus.AVAILABLE
            if (
                get_engine_status("pyqt6") == EngineStatus.AVAILABLE
                or get_engine_status("pyqt5") == EngineStatus.AVAILABLE
                or get_engine_status("pyside6") == EngineStatus.AVAILABLE
            )
            else EngineStatus.NOT_INSTALLED
        )
    if name == "gym_any":
        return (
            EngineStatus.AVAILABLE
            if (
                get_engine_status("gymnasium") == EngineStatus.AVAILABLE
                or get_engine_status("gym") == EngineStatus.AVAILABLE
            )
            else EngineStatus.NOT_INSTALLED
        )

    import_name = _MODULE_MAPPING.get(name, name)
    return _probe_engine(name, import_name)


def get_engine_error(engine_name: str) -> Exception | None:
    get_engine_status(engine_name)
    return _engine_error_cache.get(engine_name.lower())


@functools.lru_cache(maxsize=128)
def is_engine_available(engine_name: str) -> bool:
    """Check if a physics engine or library is available."""
    return get_engine_status(engine_name) == EngineStatus.AVAILABLE


def reset_engine_status_cache() -> None:
    """Clear all memoised engine-availability state.

    The probe results are cached in module-level dicts and an
    ``lru_cache`` on :func:`is_engine_available`. That memoisation is correct
    for production (engine availability does not change at runtime) but is a
    hazard under test: a fixture that injects a ``MagicMock`` into
    ``sys.modules`` for one test would otherwise leave a stale ``AVAILABLE``
    entry that poisons every later test. Test fixtures call this to guarantee
    a clean slate between tests.

    Postcondition: subsequent availability queries re-probe ``sys.modules``.
    """
    _engine_status_cache.clear()
    _engine_error_cache.clear()
    _dependency_presence_cache.clear()
    is_engine_available.cache_clear()


def get_available_engines() -> list[str]:
    """Get list of all available physics engines.

    Returns:
        List of engine names that are importable.
    """
    return [name for name in _MODULE_MAPPING if is_engine_available(name)]


def get_unavailable_engines() -> list[str]:
    """Get list of unavailable physics engines.

    Returns:
        List of engine names that are not importable.
    """
    return [name for name in _MODULE_MAPPING if not is_engine_available(name)]


def __getattr__(name: str) -> Any:
    """Lazy evaluation of _AVAILABLE variables without evaluating missing ones prematurely."""
    if name.endswith("_AVAILABLE"):
        engine = name[:-10].lower()
        if engine in _MODULE_MAPPING or engine in (
            "parquet",
            "c3d_any",
            "qt",
            "gym_any",
        ):
            return is_engine_available(engine)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


def require_engine(engine_name: str, reason: str | None = None) -> Callable[[F], F]:
    """Decorator to skip test/function if engine is not available."""
    if not (engine_name is not None):
        raise ValueError("engine_name must be provided")

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            status = get_engine_status(engine_name)
            if status != EngineStatus.AVAILABLE:
                msg = reason or f"{engine_name} not available ({status.value})"
                if status == EngineStatus.BROKEN:
                    err = get_engine_error(engine_name)
                    msg += f" - {err}"
                try:
                    import pytest

                    pytest.skip(msg)
                except ImportError:
                    raise ImportError(
                        f"Required engine '{engine_name}' is not available. {msg}"
                    ) from None
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def skip_if_unavailable(engine_name: str) -> Any:
    """Create a pytest skip marker for unavailable engines."""
    try:
        import pytest

        status = get_engine_status(engine_name)
        msg = f"{engine_name} not installed/broken ({status.value})"
        if status == EngineStatus.BROKEN:
            err = get_engine_error(engine_name)
            msg += f" - {err}"

        return pytest.mark.skipif(
            status != EngineStatus.AVAILABLE,
            reason=msg,
        )
    except ImportError:
        raise ImportError(
            "pytest is required for skip_if_unavailable. "
            "Use require_engine decorator instead."
        ) from None
