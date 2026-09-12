"""Load the actual exported URDF, then restore mandatory native semantics."""

import hashlib
import json
from typing import Any

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)


def validate_native_urdf_bundle(
    urdf_bytes: bytes, sidecar_bytes: bytes, model_bytes: bytes
) -> dict[str, Any]:
    """Reject mismatched source identity or lost closure/frame semantics."""
    meta, spec = json.loads(sidecar_bytes), json.loads(model_bytes)
    if meta.get("schema_version") != 1 or meta.get("requires_sidecar") is not True:
        raise ValueError("Mandatory native sidecar is missing or unsupported")
    if meta.get("urdf_sha256") != hashlib.sha256(urdf_bytes).hexdigest():
        raise ValueError("URDF bytes differ from sidecar identity")
    if meta.get("model_sha256") != hashlib.sha256(model_bytes).hexdigest():
        raise ValueError("Native model bytes differ from sidecar identity")
    if meta.get("limit_semantics") != "restore-unbounded-before-dynamics":
        raise ValueError("Unsupported native limit semantics")
    for key, source in (
        ("closure", "closure"),
        ("coordinate_order", "coordinate_order"),
        ("native_joints", "joints"),
        ("gravity_m_s2", "gravity_m_s2"),
    ):
        if key not in meta or meta[key] != spec[source]:
            raise ValueError(f"Native sidecar {key} differs from source")
    for key, source in (("body_links", "bodies"), ("frame_links", "frames")):
        mapping = meta.get(key)
        if (
            not isinstance(mapping, dict)
            or set(mapping) != {row["name"] for row in spec[source]}
            or any(
                not isinstance(value, str) or not value for value in mapping.values()
            )
            or len(set(mapping.values())) != len(mapping)
        ):
            raise ValueError(f"Invalid native {key}")
    return meta


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
