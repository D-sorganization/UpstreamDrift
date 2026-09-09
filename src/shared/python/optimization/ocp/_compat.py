"""bioptim availability probe and the pip-only import shims.

bioptim (``pyomeca/bioptim``) is conda-forge-only and its core touches two
things a pip-first, headless checkout does not have:

- ``biorbd_casadi`` -- used in exactly two non-model places at the pinned
  release: a type hint in ``models/protocols/holonomic_biomodel.py`` and a
  version string in ``optimization/optimal_control_program.py``.
- ``tkinter`` -- imported at module scope by ``gui/plot.py`` and used only to
  read the screen size when a plot window is opened.
- ``matplotlib.cm.get_cmap`` -- imported by ``gui/ipopt_output_plot.py``;
  removed in matplotlib 3.9 in favour of ``matplotlib.colormaps``.

:func:`require_bioptim` installs a minimal stand-in for each *only when the
real module is not importable*, then imports bioptim. Both shims disappear
once the upstream PR (#9761) that makes those imports optional lands and the
pin is bumped. Nothing here imports bioptim at module import time.
"""

from __future__ import annotations

import sys
import types
from importlib import import_module
from importlib.machinery import PathFinder
from importlib.util import find_spec
from typing import Any
from unittest.mock import Mock

__all__ = [
    "BIOPTIM_INSTALL_HINT",
    "BioptimNotAvailableError",
    "bioptim_available",
    "bioptim_version",
    "install_biorbd_shim",
    "install_matplotlib_shim",
    "install_tkinter_shim",
    "require_bioptim",
]

BIOPTIM_INSTALL_HINT = (
    "bioptim is not installed. Install the bioptim extra: "
    "pip install 'upstream-drift[bioptim]' "
    "(Debian/Ubuntu CI images also need `apt-get install python3-tk`)."
)

#: ``__version__`` reported by the ``biorbd_casadi`` shim. bioptim's
#: ``check_version`` only needs something ``packaging`` can parse.
BIORBD_SHIM_VERSION = "1.12.0"


class BioptimNotAvailableError(RuntimeError):
    """Raised when bioptim is required but not importable."""


def _is_mock_module(module: Any) -> bool:
    """Whether ``module`` is a ``unittest.mock`` stand-in, not a real module."""
    return isinstance(module, Mock)


def _importable(name: str) -> bool:
    """Whether ``name`` resolves to a *real* importable distribution.

    A spec-less ``MagicMock`` in ``sys.modules`` (the unit tree's conftest
    installs one for ``casadi`` process-wide, #9771) both raises
    ``ValueError`` out of :func:`find_spec` on Python 3.14+ and shadows the
    genuine distribution behind it. Top-level names are therefore probed
    through the path finder, which consults ``sys.path`` and ignores
    ``sys.modules`` entirely; only the absence of a real distribution --
    with or without a mock in front of it -- counts as not installed.
    """
    if "." not in name:
        try:
            return PathFinder.find_spec(name) is not None
        except (ValueError, ImportError):
            return False
    try:
        return find_spec(name) is not None
    except (ValueError, ModuleNotFoundError):
        return False


def bioptim_available() -> bool:
    """Whether ``bioptim`` and ``casadi`` are really importable.

    A spec-less ``MagicMock`` in ``sys.modules`` counts as unavailable
    *only when no real distribution sits behind it* (#9771): the unit
    tree's conftest poisons ``casadi`` process-wide, which must not make
    a co-run lane skip every bioptim test on a machine that has the stack
    installed.
    """
    return _importable("casadi") and _importable("bioptim")


class _PermissiveDummy:
    """Stands in for any ``biorbd_casadi`` attribute bioptim touches."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> _PermissiveDummy:
        return self

    def __getattr__(self, name: str) -> _PermissiveDummy:
        return self


def install_biorbd_shim() -> bool:
    """Register a ``biorbd_casadi`` stand-in unless the real one is importable.

    Returns ``True`` when the shim was installed by this call.
    """
    if "biorbd_casadi" in sys.modules or _importable("biorbd_casadi"):
        return False
    module = types.ModuleType("biorbd_casadi")
    setattr(module, "__version__", BIORBD_SHIM_VERSION)  # noqa: B010
    setattr(module, "__getattr__", lambda name: _PermissiveDummy)  # noqa: B010
    setattr(module, "__ud_shim__", True)  # noqa: B010
    sys.modules["biorbd_casadi"] = module
    return True


class _HeadlessTk:
    """Enough of ``tkinter.Tk`` for bioptim's screen-size lookup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def winfo_screenwidth(self) -> int:
        return 1920

    def winfo_screenheight(self) -> int:
        return 1080

    def destroy(self) -> None:
        pass


def install_tkinter_shim() -> bool:
    """Register a headless ``tkinter`` stand-in unless the real one imports.

    Returns ``True`` when the shim was installed by this call.
    """
    if "tkinter" in sys.modules or _importable("tkinter"):
        return False
    module = types.ModuleType("tkinter")
    setattr(module, "Tk", _HeadlessTk)  # noqa: B010
    setattr(module, "__ud_shim__", True)  # noqa: B010
    sys.modules["tkinter"] = module
    return True


def install_matplotlib_shim() -> bool:
    """Restore ``matplotlib.cm.get_cmap`` on matplotlib >= 3.9.

    Returns ``True`` when the alias was installed by this call.
    """
    try:
        import matplotlib.cm as cm
    except ImportError:  # pragma: no cover - matplotlib is a bioptim dependency
        return False
    if hasattr(cm, "get_cmap"):
        return False
    from matplotlib import colormaps

    setattr(cm, "get_cmap", colormaps.get_cmap)  # noqa: B010
    return True


def require_bioptim() -> Any:
    """Import and return ``bioptim``, raising with an install hint if absent.

    Postcondition: ``sys.modules`` carries ``biorbd_casadi`` and ``tkinter``
    (real or shim), ``matplotlib.cm.get_cmap`` resolves, and ``bioptim``
    imported without touching conda.

    Spec-less mocks for ``casadi``/``bioptim`` (the unit tree's conftest
    poisons ``casadi`` process-wide, #9771) are evicted before the import so
    bioptim binds the genuine modules, and left evicted: the unit tree's
    autouse fixture reinstalls its own mocks before each of its tests, while
    the lazy ``import casadi`` inside the ocp tests must see the real one.
    """
    if not bioptim_available():
        raise BioptimNotAvailableError(BIOPTIM_INSTALL_HINT)
    for name in ("casadi", "bioptim", "biorbd_casadi"):
        if _is_mock_module(sys.modules.get(name)):
            del sys.modules[name]
    install_biorbd_shim()
    install_tkinter_shim()
    install_matplotlib_shim()
    return import_module("bioptim")


def bioptim_version() -> str | None:
    """Installed bioptim version string, or ``None`` when unavailable."""
    if not bioptim_available():
        return None
    module = require_bioptim()
    return str(getattr(module, "__version__", "unknown"))
