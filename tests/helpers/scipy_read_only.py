"""Emulate SciPy < 1.15 ``Rotation`` constructors that reject read-only arrays.

The project supports ``scipy>=1.13.1``. Before SciPy 1.15 the Cython
``Rotation.from_matrix`` (and its sibling constructors) bound the input to a
writable memoryview, so a read-only NumPy array raised
``ValueError: buffer source array is read-only``. Newer SciPy accepts such
arrays, so a test run on a current SciPy cannot see the defect. The segmental
spine contracts freeze their arrays (``setflags(write=False)``), which made the
cross-engine lane (SciPy 1.14.1) fail while local runs passed.

:func:`strict_rotation_constructors` patches the constructors to raise exactly
as the old SciPy did, so a test reproduces the CI behaviour on any SciPy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

READ_ONLY_MESSAGE = "buffer source array is read-only"

_CONSTRUCTORS = ("from_matrix", "from_euler", "from_rotvec", "from_quat")


def _strict(original: Callable[..., Any], name: str) -> Callable[..., Any]:
    def constructor(*args: Any, **kwargs: Any) -> Any:
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, np.ndarray) and not arg.flags.writeable:
                raise ValueError(f"{READ_ONLY_MESSAGE} (Rotation.{name})")
        return original(*args, **kwargs)

    return constructor


def strict_rotation_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``Rotation`` constructors reject read-only ndarray arguments.

    Preconditions:
        ``monkeypatch`` is a pytest ``MonkeyPatch``; the patch is undone when
        it is.

    Raises:
        TypeError: ``monkeypatch`` is not a ``pytest.MonkeyPatch``.
    """
    if not isinstance(monkeypatch, pytest.MonkeyPatch):
        raise TypeError(
            f"monkeypatch must be a pytest.MonkeyPatch, got {type(monkeypatch).__name__}"
        )
    for name in _CONSTRUCTORS:
        original = getattr(Rotation, name)
        monkeypatch.setattr(Rotation, name, staticmethod(_strict(original, name)))
