"""Moco weld selection keeps free joints free (#13042 review)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    _moco_weld_joints,
)

pytestmark = pytest.mark.unit


@dataclass
class _Coord:
    locked: bool

    def get_locked(self) -> bool:
        return self.locked


@dataclass
class _Joint:
    name: str
    coords: tuple[_Coord, ...]

    def getName(self) -> str:  # noqa: N802 - OpenSim API
        return self.name

    def numCoordinates(self) -> int:  # noqa: N802 - OpenSim API
        return len(self.coords)

    def get_coordinates(self, k: int) -> _Coord:
        return self.coords[k]


class _JointSet:
    def __init__(self, joints: list[_Joint]) -> None:
        self._joints = joints

    def getSize(self) -> int:  # noqa: N802 - OpenSim API
        return len(self._joints)

    def get(self, i: int) -> _Joint:
        return self._joints[i]


class _Model:
    def __init__(self, joints: list[_Joint]) -> None:
        self._set = _JointSet(joints)

    def getJointSet(self) -> _JointSet:  # noqa: N802 - OpenSim API
        return self._set


def test_welds_mtp_and_fully_locked_joints_only() -> None:
    model = _Model(
        [
            _Joint("mtp_r", (_Coord(False),)),
            _Joint("subtalar_r", (_Coord(True),)),
            _Joint("subtalar_l", (_Coord(False),)),
            _Joint("radius_hand_r", (_Coord(True), _Coord(False))),
            _Joint("ground_pelvis", ()),
        ]
    )
    assert _moco_weld_joints(model) == ("mtp_r", "mtp_l", "subtalar_r")
