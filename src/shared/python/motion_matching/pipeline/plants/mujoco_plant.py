"""MuJoCo MatchingPlant implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.pipeline.plant import integrate_euler_step

if TYPE_CHECKING:
    from src.engines.physics_engines.mujoco.python.full_body_ik import (
        FullBodyMarkerKinematics,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )


class MujocoMatchingPlant:
    """MuJoCo implementation of MatchingPlant."""

    def __init__(self, spec: bytes | Mapping[str, Any]) -> None:
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        if isinstance(spec, bytes):
            self._spec_bytes = spec
        else:
            self._spec_bytes = json.dumps(spec).encode("utf-8")
        self.adapter: NativeMujocoFullBodyModel = NativeMujocoFullBodyModel(
            self._spec_bytes
        )

    @property
    def engine_name(self) -> str:
        return "mujoco"

    @property
    def plant_sha(self) -> str:
        return str(self.adapter.model_sha256)

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return tuple(self.adapter.coordinate_order)

    @property
    def ground_plane(self) -> GroundPlane:
        return self.adapter.ground_plane

    def create_ik(
        self, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> BaseFullBodyIK:
        from src.engines.physics_engines.mujoco.python.full_body_ik import (
            FullBodyMarkerKinematics,
        )

        ordered = {
            label: (
                attachments[label][0],
                (
                    float(attachments[label][1][0]),
                    float(attachments[label][1][1]),
                    float(attachments[label][1][2]),
                ),
            )
            for label in attachments
        }
        return FullBodyMarkerKinematics(self.adapter, ordered)

    def frame_poses(
        self, mapping: Mapping[str, tuple[str, Sequence[float]]], q: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        ik = self.create_ik(mapping)
        raw_poses = ik.pose_fn(q)
        return {b: (pose[0], pose[1]) for b, pose in raw_poses.items()}

    def marker_positions(
        self, q: np.ndarray, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> np.ndarray:
        ik = self.create_ik(attachments)
        ik._set(q)
        return np.asarray(ik._positions(), dtype=float)

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Mapping[str, float]:
        return self.adapter.accelerations(coordinates, rates, primitive_efforts)

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
    ) -> Any:
        return self.adapter.evaluate_contact_samples(coordinates, rates)

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        ik = self.create_ik({})
        return np.asarray(ik.closure_residuals(q), dtype=float)

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return integrate_euler_step(
            self.accelerations, self.coordinate_order, q, v, tau, dt
        )
