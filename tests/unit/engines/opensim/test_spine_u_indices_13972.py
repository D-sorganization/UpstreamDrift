"""Tests for coordinate unit-speed probing in spinal inverse dynamics (UD-39, #13972).

Under Epic #12735 (SPN: Segmental Spine Models in OpenSim),
``_u_indices`` maps each model coordinate to its mobility (speed) index
in the generalized velocity vector ``u``.
"""

from __future__ import annotations

import pytest

from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    _u_indices,
)

pytestmark = pytest.mark.unit


class _MockCoord:
    def __init__(self, name: str, speed_idx: int, jitter: float = 0.0) -> None:
        self._name = name
        self._speed_idx = speed_idx
        self._jitter = jitter

    def getName(self) -> str:
        return self._name

    def setSpeedValue(self, state: _MockState, val: float) -> None:
        state.updU().set(self._speed_idx, val + self._jitter)


class _MockCoordinateSet:
    def __init__(self, coords: list[_MockCoord]) -> None:
        self._coords = coords
        self._by_name = {c._name: c for c in coords}

    def getSize(self) -> int:
        return len(self._coords)

    def get(self, idx_or_name: int | str) -> _MockCoord:
        if isinstance(idx_or_name, int):
            return self._coords[idx_or_name]
        return self._by_name[idx_or_name]


class _MockVector:
    def __init__(self, size: int) -> None:
        self._data = [0.0] * size

    def set(self, idx: int, val: float) -> None:
        self._data[idx] = float(val)

    def get(self, idx: int) -> float:
        return self._data[idx]


class _MockState:
    def __init__(self, nu: int) -> None:
        self._u = _MockVector(nu)

    def getNU(self) -> int:
        return len(self._u._data)

    def updU(self) -> _MockVector:
        return self._u

    def getU(self) -> _MockVector:
        return self._u


class _MockModel:
    def __init__(self, coords: list[_MockCoord]) -> None:
        self._coord_set = _MockCoordinateSet(coords)

    def getCoordinateSet(self) -> _MockCoordinateSet:
        return self._coord_set


def test_ud_39_contract_documented() -> None:
    """_u_indices must document the coordinate unit-speed probing contract (UD-39, #13972)."""
    doc = _u_indices.__doc__ or ""
    assert "13972" in doc, "_u_indices must cite Issue #13972"
    assert "12735" in doc, "_u_indices must cite Epic #12735"


def test_u_indices_standard_mapping() -> None:
    """_u_indices maps each coordinate to its isolated speed index."""
    coords = [
        _MockCoord("lumbar_extension", 0),
        _MockCoord("lumbar_bending", 1),
        _MockCoord("lumbar_rotation", 2),
    ]
    model = _MockModel(coords)
    state = _MockState(3)

    mapping = _u_indices(model, state)
    assert mapping == {
        "lumbar_extension": 0,
        "lumbar_bending": 1,
        "lumbar_rotation": 2,
    }
    # State mobilities must be restored to zero after probing
    for i in range(state.getNU()):
        assert state.getU().get(i) == 0.0


def test_u_indices_numerical_tolerance() -> None:
    """_u_indices is robust to floating-point jitter around 1.0."""
    coords = [
        _MockCoord("q0", 0, jitter=1e-8),
        _MockCoord("q1", 1, jitter=-1e-8),
    ]
    model = _MockModel(coords)
    state = _MockState(2)

    mapping = _u_indices(model, state)
    assert mapping == {"q0": 0, "q1": 1}


def test_u_indices_ambiguous_or_unmapped_raises() -> None:
    """Invariant error when coordinate maps to multiple or zero speeds."""
    # Coordinate that doesn't set any speed
    class _DeadCoord(_MockCoord):
        def setSpeedValue(self, state: _MockState, val: float) -> None:
            pass

    model_dead = _MockModel([_DeadCoord("dead_coord", 0)])
    state = _MockState(1)
    with pytest.raises(Exception, match="maps to"):
        _u_indices(model_dead, state)
