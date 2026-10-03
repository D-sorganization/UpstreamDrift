"""MuJoCo MatchingPlant implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.pipeline.plant import integrate_euler_step

if TYPE_CHECKING:
    from src.shared.python.motion_matching.full_body_forward_dynamics import (
        FullBodySimulator,
    )
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
    def coordinate_units(self) -> tuple[str, ...]:
        """SI units verified from named compiled scalar joints, in declared order."""
        return self.adapter.coordinate_units

    @property
    def ground_plane(self) -> GroundPlane:
        return self.adapter.ground_plane

    def create_forward_simulator(self) -> FullBodySimulator:
        """Reuse native mass, gravity, contact and closure dynamics."""
        from src.shared.python.motion_matching.full_body_forward_dynamics import (
            FullBodySimulator,
        )

        return FullBodySimulator(self.adapter)

    def create_ik(
        self,
        attachments: Mapping[str, tuple[str, Sequence[float]]],
        *,
        ik_backend: str = "lm",
    ) -> FullBodyMarkerKinematics:
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
        if ik_backend == "mujoco-minimize":
            from src.engines.physics_engines.mujoco.python.ik_minimize import (
                MinimizeMarkerKinematics,
            )

            return MinimizeMarkerKinematics(self.adapter, ordered)
        if ik_backend != "lm":
            raise ValueError(
                f"Unknown MuJoCo IK backend {ik_backend!r}; expected 'lm' or 'mujoco-minimize'"
            )
        return FullBodyMarkerKinematics(self.adapter, ordered)

    def frame_poses(
        self, mapping: Mapping[str, tuple[str, Sequence[float]]], q: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        q_array = np.asarray(q, dtype=float)
        if (
            q_array.shape != (len(self.coordinate_order),)
            or not np.isfinite(q_array).all()
        ):
            raise ValueError("Coordinates must be a finite vector of model size")
        ik = self.create_ik(mapping)
        return ik.body_poses(q_array, tuple(body for body, _ in mapping.values()))

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
        """Evaluate grip separation without requiring observation markers."""
        q_array = np.asarray(q, dtype=float)
        order = self.coordinate_order
        if q_array.shape != (len(order),) or not np.isfinite(q_array).all():
            raise ValueError("Coordinates must be a finite vector of model size")
        coordinates = dict(zip(order, q_array, strict=True))
        return self.adapter.kinematic_closure_residuals(coordinates)

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return integrate_euler_step(
            self.accelerations, self.coordinate_order, q, v, tau, dt
        )
