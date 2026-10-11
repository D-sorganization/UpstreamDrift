"""Skip real-engine tests when the engine module is a test double (issue #12204).

``tests/unit/conftest.py`` installs a ``MagicMock`` in ``sys.modules`` for
``pydrake``/``pydrake.all``, ``pinocchio``, ``pinocchio.casadi`` and
``casadi`` when they are not installed, so ``pytest.importorskip`` returns
the mock instead of skipping. Other tests leave spec-less stub modules
behind. A test that needs the real engine calls ``real_module`` instead::

    from tests.support.real_engines import real_module

    pin = real_module("pinocchio")
"""

from __future__ import annotations

import sys
import types

import pytest

__all__ = ["is_test_double", "real_module"]


def is_test_double(module: object) -> bool:
    """Return True if ``module`` is a mock or stub rather than an installed module.

    Real modules are ``ModuleType`` instances loaded from a file (``__file__``
    set); built-in modules are real too. ``MagicMock`` and hand-built
    ``types.ModuleType`` stubs fail one of those checks.
    """
    if not isinstance(module, types.ModuleType):
        return True
    return not hasattr(module, "__file__") and module.__name__ not in (
        sys.builtin_module_names
    )


def real_module(name: str) -> types.ModuleType:
    """Import ``name`` or skip; skip too when it (or its root package) is a test double.

    Raises ``ValueError`` for an empty ``name``. Postcondition: the returned
    module and its top-level package are real installed modules.
    """
    if not name:
        raise ValueError("real_module needs a module name")
    module = pytest.importorskip(name)
    root = sys.modules.get(name.split(".", 1)[0], module)
    if is_test_double(module) or is_test_double(root):
        pytest.skip(f"{name} in sys.modules is a test double, not the real package")
    return module
