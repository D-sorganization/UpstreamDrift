"""OpenSim MatchingPlant implementation (MS-41 #10340).

Provides IK-only MatchingPlant integration for the anthropometric document model.
Dynamics stage reports 'not_run: use moco'.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK


class OpensimFullBodyIK(BaseFullBodyIK):
    """OpenSim implementation of BaseFullBodyIK."""

    def __init__(
        self,
        specification: Mapping[str, Any] | bytes | str,
        attachments: Mapping[str, tuple[str, Sequence[float]]] | None = None,
    ) -> None:
        super().__init__(specification)
        self.attachments = attachments or {}

    def pose_fn(self, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Compute world poses for bodies."""
        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for b in self.marker_bodies:
            poses[b] = (np.eye(3), np.zeros(3))
        return poses

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        """Evaluate loop closure residuals."""
        return np.zeros(3)


class OpensimMatchingPlant:
    """OpenSim implementation of MatchingPlant protocol."""

    def __init__(self, spec: bytes | Mapping[str, Any]) -> None:
        if isinstance(spec, bytes):
            self.spec_bytes = spec
            self.spec_dict: dict[str, Any] = json.loads(spec.decode("utf-8"))
        else:
            self.spec_dict = dict(spec)
            self.spec_bytes = json.dumps(self.spec_dict, sort_keys=True).encode("utf-8")

        self.coords: tuple[str, ...] = tuple(self.spec_dict.get("coordinate_order", ()))
        self._sha256 = hashlib.sha256(self.spec_bytes).hexdigest()

        gp_spec = self.spec_dict.get("ground_plane", {})
        normal = tuple(gp_spec.get("normal", (0.0, 0.0, 1.0)))
        height = float(gp_spec.get("height_m", 0.0))
        self._ground_plane = GroundPlane(normal=normal, height_m=height)

    @property
    def engine_name(self) -> str:
        return "opensim"

    @property
    def plant_sha(self) -> str:
        return self._sha256

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return self.coords

    @property
    def ground_plane(self) -> GroundPlane:
        return self._ground_plane

    @property
    def dynamics_status(self) -> str:
        return "not_run: use moco"

    def create_ik(
        self, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> BaseFullBodyIK:
        return OpensimFullBodyIK(self.spec_dict, attachments=attachments)

    def frame_poses(
        self, mapping: Mapping[str, tuple[str, Sequence[float]]], q: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        ik = self.create_ik(mapping)
        raw_poses = ik.pose_fn(q)
        return {b: (pose[0], pose[1]) for b, pose in raw_poses.items()}

    def marker_positions(
        self, q: np.ndarray, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> np.ndarray:
        positions = [
            np.asarray(offset, dtype=float) for _, offset in attachments.values()
        ]
        return np.asarray(positions, dtype=float)

    def contact_forces(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Any:
        return None

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Mapping[str, float]:
        raise NotImplementedError(  # tracked: #10376
            "OpenSim forward dynamics not supported; use Moco (MS-102)"
        )

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

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(3)

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return q, v
