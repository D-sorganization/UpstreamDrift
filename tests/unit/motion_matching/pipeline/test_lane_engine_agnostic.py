"""Unit tests for engine-agnostic Lane and pipeline execution."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.marker_calibration import Pose
from src.shared.python.motion_matching.pipeline.plant import MatchingPlant
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


class FakeFullBodyIK(BaseFullBodyIK):
    """Fake IK for testing."""

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
        return {b: (np.eye(3), np.zeros(3)) for b in self.marker_bodies}

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def _set(self, q: np.ndarray) -> None:
        pass

    def _positions(self) -> np.ndarray:
        return np.zeros((len(self.labels), 3), dtype=float)

    def _marker_jacobian(self, positions: np.ndarray) -> np.ndarray:
        return np.zeros((len(self.labels) * 3, len(self.coordinate_order)), dtype=float)


class FakePlant:
    """Minimal fake plant implementing MatchingPlant."""

    def __init__(self, ground_height: float = 0.0) -> None:
        self._coords = ("joint_1", "joint_2", "joint_3")
        self.ground_plane = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=ground_height)

    @property
    def engine_name(self) -> str:
        return "fake"

    @property
    def plant_sha(self) -> str:
        return "sha_fake_plant"

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return self._coords

    def create_ik(
        self,
        attachments: Mapping[str, tuple[str, Sequence[float]]],
        *,
        ik_backend: str = "lm",
    ) -> BaseFullBodyIK:
        del ik_backend  # FakePlant ignores backend selection
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
        return {}

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return q + v * dt, v


def test_lane_with_fake_plant_runs_kinematics_without_engine_imports() -> None:
    from src.shared.python.motion_matching.pipeline.lane import Lane

    labels = ("RAnkleOut", "LAnkleOut", "RToeIn", "RToeOut", "LToeIn", "LToeOut")
    points = np.zeros((10, len(labels), 3), dtype=float)
    valid = np.ones((10, len(labels)), dtype=bool)
    time_s = np.linspace(0.0, 0.1, 10)
    capture = TourCapture(
        time_s=time_s,
        labels=labels,
        points_m=points,
        valid=valid,
    )

    lane = Lane(labels=labels, capture=capture)
    fake_plant = FakePlant(ground_height=lane.ground.height_m)
    lane.plant = fake_plant
    assert lane.plant is fake_plant
    assert lane.frames == 10

    attachments = dict.fromkeys(labels, ("body_1", (0.0, 0.0, 0.0)))
    adapter, kin = lane.kinematics(b"{}", attachments)
    assert isinstance(kin, BaseFullBodyIK)


def test_pipeline_cli_parser_defaults() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([])
    assert args.engine == "mujoco"
    assert args.backend == "mujoco"
    assert args.capture == "driver"


def test_run_ground_support_shim_preserves_exports() -> None:
    from docs.development.full_body_models.evidence.ground_support import (
        run_ground_support as shim,
    )

    assert hasattr(shim, "main")
    assert hasattr(shim, "BUILD_RECEIPT")
    assert hasattr(shim, "SPEC")
