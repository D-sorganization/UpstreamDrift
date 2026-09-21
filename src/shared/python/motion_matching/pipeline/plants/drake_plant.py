"""Drake MatchingPlant implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK

if TYPE_CHECKING:
    from src.engines.physics_engines.drake.python.full_body_ik import DrakeFullBodyIK
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )


class DrakeMatchingPlant:
    """Drake implementation of MatchingPlant."""

    def __init__(self, spec: bytes | Mapping[str, Any]) -> None:
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        if isinstance(spec, bytes):
            self._spec_bytes = spec
            self.spec_dict: dict[str, Any] = json.loads(spec.decode("utf-8"))
        else:
            self.spec_dict = dict(spec)
            self._spec_bytes = json.dumps(spec).encode("utf-8")
        self.model: FullBodyDrakeModel = FullBodyDrakeModel(self.spec_dict)
        self.adapter = self.model
        self.upper_body_coordinates: int = int(
            self.spec_dict.get("upper_body_counts", {}).get("coordinates", 0)
        )
        self.model.upper_body_coordinates = self.upper_body_coordinates

    @property
    def engine_name(self) -> str:
        return "drake"

    @property
    def plant_sha(self) -> str:
        return str(self.model.model_sha256)

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return tuple(self.model.names)

    @property
    def ground_plane(self) -> GroundPlane:
        return self.model.ground_plane

    def create_ik(
        self, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> BaseFullBodyIK:
        from src.engines.physics_engines.drake.python.full_body_ik import (
            DrakeFullBodyIK,
        )

        return DrakeFullBodyIK(self.spec_dict, attachments=attachments)

    def frame_poses(
        self, mapping: Mapping[str, tuple[str, Sequence[float]]], q: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        ik = self.create_ik(mapping)
        raw_poses = ik.pose_fn(q)
        return {b: (pose[0], pose[1]) for b, pose in raw_poses.items()}

    def marker_positions(
        self, q: np.ndarray, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> np.ndarray:
        return self.model.marker_positions(q, attachments)

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Mapping[str, float]:
        return self.model.accelerations(coordinates, rates, primitive_efforts)

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
        return self.model.contact_forces(coordinates, rates)

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        coords = self.coordinate_order
        q_dict = {c: float(q[i]) for i, c in enumerate(coords)}
        zero_rates = dict.fromkeys(coords, 0.0)
        pos_res, _ = self.model.closure_residuals(q_dict, zero_rates)
        return np.asarray(pos_res[:3], dtype=float)

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.model.step(q, v, tau, dt)

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Execute Drake native full-body trajectory fitting (MS-30)."""
        from src.engines.physics_engines.drake.python.full_body_fit import (
            fit_full_body_drake,
        )

        return fit_full_body_drake(self.spec_dict, *args, **kwargs)
