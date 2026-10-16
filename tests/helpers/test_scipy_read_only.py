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


def test_read_only_euler_angles_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict_rotation_constructors(monkeypatch)
    with pytest.raises(ValueError, match=READ_ONLY_MESSAGE):
        Rotation.from_euler("ZXY", _frozen(np.zeros(3)))


def test_writable_input_still_converts(monkeypatch: pytest.MonkeyPatch) -> None:
    strict_rotation_constructors(monkeypatch)
    np.testing.assert_allclose(Rotation.from_matrix(np.eye(3)).as_matrix(), np.eye(3))


def test_patch_is_undone_with_the_monkeypatch() -> None:
    before = Rotation.__dict__["from_matrix"]
    patch = pytest.MonkeyPatch()
    strict_rotation_constructors(patch)
    assert Rotation.__dict__["from_matrix"] is not before
    patch.undo()
    assert Rotation.__dict__["from_matrix"] is before


def test_non_monkeypatch_argument_is_rejected() -> None:
    with pytest.raises(TypeError, match="MonkeyPatch"):
        strict_rotation_constructors(object())  # type: ignore[arg-type]
