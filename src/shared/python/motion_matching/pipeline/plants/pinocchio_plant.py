"""Pinocchio MatchingPlant implementation (MS-10 / MS-14).

Wraps ``FullBodyPinocchioModel`` (``constraintDynamics`` weld closure + shared
contact law). Pink is exposed as a ConstrainedIKBackend via
``create_constrained_ik``; ``pinocchio`` / ``pink`` imports stay lazy (LoD).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.pipeline.plant import (
    compute_attachment_marker_positions,
    integrate_euler_step,
)

if TYPE_CHECKING:
    from src.engines.physics_engines.pinocchio.python.full_body_ik import (
        PinocchioFullBodyIK,
    )
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )
    from src.shared.python.motion_matching.constrained_ik import ConstrainedIKBackend

_SUPPORTED_IK_BACKENDS = frozenset({"lm", "pink"})


class PinocchioMatchingPlant:
    """Pinocchio implementation of MatchingPlant."""

    def __init__(self, spec: bytes | Mapping[str, Any]) -> None:
        from src.engines.physics_engines.pinocchio.python.native_model import (
            FullBodyPinocchioModel,
        )

        if isinstance(spec, bytes):
            self.spec_dict: dict[str, Any] = json.loads(spec.decode("utf-8"))
        else:
            self.spec_dict = dict(spec)
        self.model: FullBodyPinocchioModel = FullBodyPinocchioModel(self.spec_dict)
        self._sha = canonical_sha256(self.spec_dict)

    @property
    def engine_name(self) -> str:
        return "pinocchio"

    @property
    def plant_sha(self) -> str:
        return self._sha

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return tuple(self.model.coordinate_order)

    @property
    def ground_plane(self) -> GroundPlane:
        return self.model.ground

    def create_ik(
        self,
        attachments: Mapping[str, tuple[str, Sequence[float]]],
        *,
        ik_backend: str = "lm",
    ) -> BaseFullBodyIK:
        if ik_backend not in _SUPPORTED_IK_BACKENDS:
            raise ValueError(
                f"Unknown Pinocchio IK backend {ik_backend!r}; "
                f"expected one of {sorted(_SUPPORTED_IK_BACKENDS)}"
            )
        if ik_backend == "pink":
            raise ValueError(
                "Pink is a trajectory ConstrainedIKBackend; call "
                "create_constrained_ik() or run the pipeline with --backend pink"
            )
        from src.engines.physics_engines.pinocchio.python.full_body_ik import (
            PinocchioFullBodyIK,
        )

        spec_copy = dict(self.spec_dict)
        spec_copy["marker_attachments"] = {
            label: {"body": body, "offset": list(offset)}
            for label, (body, offset) in attachments.items()
        }
        ik = PinocchioFullBodyIK(spec_copy)
        ik.labels = tuple(attachments.keys())
        return ik

    def create_constrained_ik(self) -> ConstrainedIKBackend:
        """Return the Pink ConstrainedIKBackend bound to this plant's document.

        Fails closed when Pink or Pinocchio cannot be imported.
        """
        from src.engines.physics_engines.pinocchio.python.pink_trajectory import (
            PinkTrajectoryService,
        )

        return PinkTrajectoryService(self.spec_dict, model=self.model)

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
        return compute_attachment_marker_positions(ik.pose_fn(q), attachments)

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
        return self.model.acceleration_derivatives(
            coordinates, rates, primitive_efforts
        )

    def contact_effort_derivatives(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Any:
        return self.model.contact_effort_derivatives(coordinates, rates)

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
        return integrate_euler_step(
            self.accelerations, self.coordinate_order, q, v, tau, dt
        )
