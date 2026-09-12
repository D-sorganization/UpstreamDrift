"""Load the actual exported URDF, then restore mandatory native semantics."""

from typing import Any

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.motion_matching.native_urdf_contract import (
    validate_native_urdf_bundle as validate_native_urdf_bundle,
)


class NativeUrdfModel(NativePinocchioModel):
    """Share replay operations while obtaining all bodies/inertias from URDF."""

    def __init__(
        self, urdf_bytes: bytes, sidecar_bytes: bytes, model_bytes: bytes
    ) -> None:
        meta = validate_native_urdf_bundle(urdf_bytes, sidecar_bytes, model_bytes)
        import pinocchio as pin

        self._pin = pin
        # Pinocchio's extension object exposes fields absent from local stubs.
        self.model: Any = pin.buildModelFromXML(urdf_bytes.decode("utf-8"))
        self.model.gravity.linear[:] = meta["gravity_m_s2"]
        self.model.lowerPositionLimit[:] = -np.inf
        self.model.upperPositionLimit[:] = np.inf
        self.model.effortLimit[:] = np.inf
        self.model.velocityLimit[:] = np.inf
        names = meta["coordinate_order"]
        if (
            set(self.model.names[1:]) != set(names)
            or self.model.nq != len(names)
            or self.model.nv != len(names)
        ):
            raise ValueError("URDF did not preserve native scalar coordinates")
        self._coordinates, self._velocity_indices = {}, {}
        for name in names:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError("URDF primitive is not scalar")
            self._coordinates[name], self._velocity_indices[name] = (
                joint.idx_q,
                joint.idx_v,
            )

        def frame_id(name: str) -> int:
            index = self.model.getFrameId(name)
            if index >= self.model.nframes:
                raise ValueError(f"URDF frame is missing: {name}")
            return index

        self._bodies = {}
        for name, link in meta["body_links"].items():
            frame = self.model.frames[frame_id(link)]
            self._bodies[name] = (frame.parentJoint, frame.placement.copy())
        self._frames = {
            name: frame_id(link) for name, link in meta["frame_links"].items()
        }
        self._initialize_closure(meta["closure"])
