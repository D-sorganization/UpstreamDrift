"""URDF bundle validation contract for Drake full-body models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _validate_link_mapping(
    meta: Mapping[str, Any], key: str, expected_names: set[str]
) -> None:
    mapping = meta.get(key)
    if (
        not isinstance(mapping, dict)
        or set(mapping) != expected_names
        or any(not isinstance(v, str) or not v for v in mapping.values())
        or len(set(mapping.values())) != len(mapping)
    ):
        raise ValueError(f"Invalid full-body {key}")


def validate_full_body_urdf_bundle(
    urdf_bytes: bytes, sidecar_bytes: bytes, model_bytes: bytes
) -> dict[str, Any]:
    """Reject mismatched source identity or corrupted sidecar semantics."""
    meta = json.loads(sidecar_bytes)
    spec = json.loads(model_bytes)

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
            raise ValueError(f"Full-body sidecar {key} differs from source")

    _validate_link_mapping(
        meta, "body_links", {body["name"] for body in spec["bodies"]}
    )
    _validate_link_mapping(
        meta, "frame_links", {frame["name"] for frame in spec["frames"]}
    )
    _validate_link_mapping(
        meta,
        "contact_links",
        {sphere["name"] for sphere in spec["contact"]["spheres"]},
    )

    return meta
