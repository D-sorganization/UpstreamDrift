"""Drake MatchingPlant implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

import numpy as np

from src.engines.physics_engines.drake.python.full_body_ik import DrakeFullBodyIK
from src.engines.physics_engines.drake.python.full_body_model import (
    FullBodyDrakeModel,
)
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK


class DrakeMatchingPlant:
    """Drake implementation of MatchingPlant."""

    def __init__(self, spec: bytes | Mapping[str, Any]) -> None:
        if isinstance(spec, bytes):
            self.spec_dict: dict[str, Any] = json.loads(spec.decode("utf-8"))
        else:
            self.spec_dict = dict(spec)
        self.model = FullBodyDrakeModel(self.spec_dict)

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
        spec_copy = dict(self.spec_dict)
        spec_copy["marker_attachments"] = {
            label: {"body": body, "offset": list(offset)}
            for label, (body, offset) in attachments.items()
        }
        ik = DrakeFullBodyIK(spec_copy)
        ik.labels = tuple(attachments.keys())
        return ik

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
        poses = ik.pose_fn(q)
        positions = []
        for label in attachments:
            body, offset = attachments[label]
            rot, trans = poses[body]
            pos = rot @ np.asarray(offset, dtype=float) + trans
            positions.append(pos)
        return np.asarray(positions, dtype=float)

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
        coords = self.coordinate_order
        q_dict = {c: float(q[i]) for i, c in enumerate(coords)}
        v_dict = {c: float(v[i]) for i, c in enumerate(coords)}
        tau_dict = {c: float(tau[i]) for i, c in enumerate(coords)}
        acc = self.accelerations(q_dict, v_dict, tau_dict)
        a = np.array([acc[c] for c in coords], dtype=float)
        next_v = v + a * dt
        next_q = q + next_v * dt
        return next_q, next_v
