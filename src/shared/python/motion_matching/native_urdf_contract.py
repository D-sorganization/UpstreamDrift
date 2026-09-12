"""Engine-independent native URDF identity and sidecar contracts."""

import hashlib
import json
from typing import Any


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
