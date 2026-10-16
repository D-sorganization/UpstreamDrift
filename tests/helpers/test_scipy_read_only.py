"""Tests for the SciPy < 1.15 read-only ``Rotation`` emulation helper."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tests.helpers.scipy_read_only import (
    READ_ONLY_MESSAGE,
    strict_rotation_constructors,
)

pytestmark = pytest.mark.unit


def _frozen(value: np.ndarray) -> np.ndarray:
    value = np.array(value, dtype=float)
    value.setflags(write=False)
    return value


def test_read_only_matrix_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    strict_rotation_constructors(monkeypatch)
    with pytest.raises(ValueError, match=READ_ONLY_MESSAGE):
        Rotation.from_matrix(_frozen(np.eye(3)))


def test_read_only_rotvec_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    strict_rotation_constructors(monkeypatch)
    with pytest.raises(ValueError, match=READ_ONLY_MESSAGE):
        Rotation.from_rotvec(_frozen(np.zeros(3)))


def test_read_only_euler_and_quat_are_accepted_as_in_scipy_1_14(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SciPy 1.14.1 accepts read-only Euler angles and quaternions; only the
    # matrix and rotation-vector constructors bind a writable memoryview.
    strict_rotation_constructors(monkeypatch)
    Rotation.from_euler("ZXY", _frozen(np.zeros(3)))
    Rotation.from_quat(_frozen([0.0, 0.0, 0.0, 1.0]))


def test_writable_input_still_converts(monkeypatch: pytest.MonkeyPatch) -> None:
    strict_rotation_constructors(monkeypatch)
    np.testing.assert_allclose(Rotation.from_matrix(np.eye(3)).as_matrix(), np.eye(3))


def test_patch_is_undone_with_the_monkeypatch() -> None:
    before = Rotation.__dict__["from_matrix"]
    patch = pytest.MonkeyPatch()
    patched = strict_rotation_constructors(patch)
    assert (Rotation.__dict__["from_matrix"] is not before) is patched
    patch.undo()
    assert Rotation.__dict__["from_matrix"] is before


class _ImmutableMeta(type):
    def __setattr__(cls, name: str, value: object) -> None:
        raise TypeError(f"cannot set {name!r} attribute of immutable type")


class _ImmutableRotation(metaclass=_ImmutableMeta):
    @staticmethod
    def from_matrix(matrix: np.ndarray) -> np.ndarray:
        return matrix


def test_immutable_rotation_type_is_left_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Cython Rotation of SciPy < 1.15 is immutable and already rejects
    # read-only matrices natively, so the helper must not try to patch it.
    original = _ImmutableRotation.__dict__["from_matrix"]
    assert not strict_rotation_constructors(monkeypatch, _ImmutableRotation)
    assert _ImmutableRotation.__dict__["from_matrix"] is original


def test_non_type_rotation_argument_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="rotation_type"):
        strict_rotation_constructors(monkeypatch, object())  # type: ignore[arg-type]


def test_non_monkeypatch_argument_is_rejected() -> None:
    with pytest.raises(TypeError, match="MonkeyPatch"):
        strict_rotation_constructors(object())  # type: ignore[arg-type]
