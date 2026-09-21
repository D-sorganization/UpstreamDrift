"""Unit tests for MatchingPlant protocol and engine plant registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.marker_calibration import Pose
from src.shared.python.motion_matching.pipeline.plant import (
    EngineUnavailableError,
    MatchingPlant,
    available_engines,
    get_plant,
    register_plant,
)

pytestmark = pytest.mark.unit


class FakeFullBodyIK(BaseFullBodyIK):
    """Minimal fake IK for protocol verification."""

    def __init__(
        self,
        coordinate_order: tuple[str, ...],
        attachments: Mapping[str, tuple[str, Sequence[float]]],
    ) -> None:
        super().__init__(
            specification={"coordinate_order": list(coordinate_order)},
            coordinate_order=coordinate_order,
            labels=tuple(attachments.keys()),
        )
        self.attachments = attachments

    def pose_fn(self, q: np.ndarray) -> dict[str, Pose]:
        r = np.eye(3, dtype=float)
        t = np.zeros(3, dtype=float)
        return dict.fromkeys(self.marker_bodies, (r, t))

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def _set(self, q: np.ndarray) -> None:
        pass

    def _positions(self) -> np.ndarray:
        return np.zeros((len(self.labels), 3), dtype=float)

    def _marker_jacobian(self, positions: np.ndarray) -> np.ndarray:
        return np.zeros((len(self.labels) * 3, len(self.coordinate_order)), dtype=float)


class FakePlant:
    """Minimal 3-DOF pure-NumPy fake plant for protocol testing."""

    def __init__(self, ground_height: float = 0.0) -> None:
        self._coords = ("joint_1", "joint_2", "joint_3")
        self._engine_name = "fake"
        self._plant_sha = "sha_fake_12345"
        self.ground_plane = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=ground_height)

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @property
    def plant_sha(self) -> str:
        return self._plant_sha

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return self._coords

    def create_ik(
        self, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> BaseFullBodyIK:
        return FakeFullBodyIK(self._coords, attachments)

    def frame_poses(
        self, mapping: Mapping[str, tuple[str, Sequence[float]]], q: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {k: (np.eye(3), np.zeros(3)) for k in mapping}

    def marker_positions(
        self, q: np.ndarray, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> np.ndarray:
        return np.zeros((len(attachments), 3), dtype=float)

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Mapping[str, float]:
        return {c: float(primitive_efforts.get(c, 0.0)) for c in self._coords}

    def acceleration_derivatives(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Any:
        return None

    def contact_effort_derivatives(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Any:
        return None

    def contact_forces(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Mapping[str, float]:
        return {"heel_r": 0.0, "toe_r": 0.0}

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        next_v = v + tau * dt
        next_q = q + next_v * dt
        return next_q, next_v


def test_fake_plant_satisfies_matching_plant_protocol() -> None:
    plant = FakePlant()
    assert isinstance(plant, MatchingPlant)
    assert plant.engine_name == "fake"
    assert plant.plant_sha == "sha_fake_12345"
    assert len(plant.coordinate_order) == 3


def test_plant_registry_unknown_engine_fails_closed() -> None:
    with pytest.raises(EngineUnavailableError, match="Unknown engine 'nonexistent'"):
        get_plant("nonexistent", b"{}")


def test_plant_registry_custom_registration() -> None:
    register_plant("custom_fake", lambda spec: FakePlant())
    assert "custom_fake" in available_engines()
    plant = get_plant("custom_fake", b"{}")
    assert isinstance(plant, MatchingPlant)
    assert plant.engine_name == "fake"


def test_mujoco_plant_instantiation() -> None:
    pytest.importorskip("mujoco")
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    spec_path = root / "docs/development/full_body_models/full_body_spec_v1.json"
    plant = get_plant("mujoco", spec_path.read_bytes())
    assert isinstance(plant, MatchingPlant)
    assert plant.engine_name == "mujoco"
    assert len(plant.coordinate_order) == 41


def test_pinocchio_matching_plant_module_export() -> None:
    from src.engines.physics_engines.pinocchio.python.matching_plant import (
        PinocchioMatchingPlant,
    )

    assert PinocchioMatchingPlant is not None


def test_drake_matching_plant_module_export() -> None:
    from src.engines.physics_engines.drake.python.matching_plant import (
        DrakeMatchingPlant,
    )

    assert DrakeMatchingPlant is not None


def test_drake_plant_instantiation() -> None:
    try:
        import pydrake.all as drake_all
    except ImportError as exc:
        pytest.skip(f"pydrake not importable: {exc}")
    if type(drake_all).__module__ == "unittest.mock" or not hasattr(
        drake_all, "MultibodyPlant"
    ):
        pytest.skip("pydrake is mocked, not a real Drake installation")
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    spec_path = (
        root / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    )
    plant = get_plant("drake", spec_path.read_bytes())
    assert isinstance(plant, MatchingPlant)
    assert plant.engine_name == "drake"
    assert len(plant.coordinate_order) in (41, 44)
