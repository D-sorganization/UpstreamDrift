"""Emulate SciPy < 1.15 ``Rotation`` constructors that reject read-only arrays.

The project supports ``scipy>=1.13.1``. Before SciPy 1.15 the Cython
``Rotation.from_matrix`` and ``Rotation.from_rotvec`` bound the input to a
writable memoryview, so a read-only NumPy array raised
``ValueError: buffer source array is read-only``. ``from_euler`` and
``from_quat`` accepted it (verified against SciPy 1.14.1). Newer SciPy accepts such
arrays, so a test run on a current SciPy cannot see the defect. The segmental
spine contracts freeze their arrays (``setflags(write=False)``), which made the
cross-engine lane (SciPy 1.14.1) fail while local runs passed.

:func:`strict_rotation_constructors` patches the constructors to raise exactly
as the old SciPy did, so a test reproduces the CI behaviour on any SciPy whose ``Rotation`` type can
be patched. Where ``Rotation`` is an immutable Cython type it is left
untouched. That type is natively strict on SciPy 1.14.1, but SciPy 1.15.3 is
immutable and already accepts a read-only ``from_matrix`` input (its
``from_rotvec`` still rejects one), so callers that need the emulation must
check :func:`natively_strict` when the patch was not applied.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

READ_ONLY_MESSAGE = "buffer source array is read-only"

_CONSTRUCTORS = ("from_matrix", "from_rotvec")


def _strict(original: Callable[..., Any], name: str) -> Callable[..., Any]:
    def constructor(*args: Any, **kwargs: Any) -> Any:
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, np.ndarray) and not arg.flags.writeable:
                raise ValueError(f"{READ_ONLY_MESSAGE} (Rotation.{name})")
        return original(*args, **kwargs)

    return constructor


def _is_mutable(rotation_type: type) -> bool:
    try:
        rotation_type._ud_read_only_probe = None  # type: ignore[attr-defined]
    except TypeError:
        return False
    del rotation_type._ud_read_only_probe  # type: ignore[attr-defined]
    return True


def natively_strict(name: str, rotation_type: type = Rotation) -> bool:
    """Return whether ``rotation_type.<name>`` rejects a read-only input itself.

    Preconditions:
        ``name`` is ``"from_matrix"`` or ``"from_rotvec"``.

    Postconditions:
        ``True`` iff the unpatched constructor raises the read-only
        ``ValueError`` for a frozen identity input.

    Raises:
        ValueError: ``name`` is not one of the emulated constructors.
    """
    if name not in _CONSTRUCTORS:
        raise ValueError(f"name must be one of {_CONSTRUCTORS}, got {name!r}")
    probe = np.eye(3) if name == "from_matrix" else np.zeros(3)
    probe.setflags(write=False)
    try:
        getattr(rotation_type, name)(probe)
    except ValueError as exc:
        return READ_ONLY_MESSAGE in str(exc)
    return False


def strict_rotation_constructors(
    monkeypatch: pytest.MonkeyPatch, rotation_type: type = Rotation
) -> bool:
    """Make ``Rotation`` constructors reject read-only ndarray arguments.

    Preconditions:
        ``monkeypatch`` is a pytest ``MonkeyPatch``; the patch is undone when
        it is. ``rotation_type`` is a class (``Rotation`` by default).

    Postconditions:
        Returns ``True`` when the constructors were patched, ``False`` when
        ``rotation_type`` is immutable and was left unchanged. An unchanged
        type is not necessarily strict; see :func:`natively_strict`.

    Raises:
        TypeError: ``monkeypatch`` is not a ``pytest.MonkeyPatch`` or
            ``rotation_type`` is not a class.
    """
    if not isinstance(monkeypatch, pytest.MonkeyPatch):
        raise TypeError(
            f"monkeypatch must be a pytest.MonkeyPatch, got {type(monkeypatch).__name__}"
        )
    if not isinstance(rotation_type, type):
        raise TypeError(
            f"rotation_type must be a class, got {type(rotation_type).__name__}"
        )
    if not _is_mutable(rotation_type):
        return False
    for name in _CONSTRUCTORS:
        original = getattr(rotation_type, name)
        monkeypatch.setattr(rotation_type, name, staticmethod(_strict(original, name)))
    return True
