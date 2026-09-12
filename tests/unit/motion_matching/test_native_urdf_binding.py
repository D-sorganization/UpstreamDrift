"""Bind mandatory sidecar semantics before importing a URDF engine model."""

import hashlib
import json

import pytest

from src.engines.physics_engines.pinocchio.python.native_urdf_model import (
    validate_native_urdf_bundle,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def bundle() -> tuple:
    xml = b'<robot name="test"/>'
    spec = {
        "coordinate_order": ["x"],
        "closure": {"name": "weld"},
        "joints": [],
        "gravity_m_s2": [0, 0, -9.81],
        "bodies": [{"name": "world"}],
        "frames": [],
    }
    raw = json.dumps(spec).encode()
    meta = {
        "schema_version": 1,
        "requires_sidecar": True,
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "urdf_sha256": hashlib.sha256(xml).hexdigest(),
        "limit_semantics": "restore-unbounded-before-dynamics",
        "coordinate_order": spec["coordinate_order"],
        "closure": spec["closure"],
        "native_joints": [],
        "gravity_m_s2": spec["gravity_m_s2"],
        "body_links": {"world": "body_0"},
        "frame_links": {},
    }
    return xml, meta, raw


def test_identity_and_semantics_are_bound(bundle: tuple) -> None:
    xml, meta, raw = bundle
    result = validate_native_urdf_bundle(xml, json.dumps(meta).encode(), raw)
    assert result["coordinate_order"] == ["x"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("requires_sidecar", False),
        ("closure", {}),
        ("coordinate_order", []),
        ("limit_semantics", "use-urdf-limits"),
        ("body_links", {}),
        ("gravity_m_s2", [0, 0, 0]),
    ],
)
def test_changed_semantics_fail_before_engine_load(
    bundle: tuple, field: str, value: object
) -> None:
    xml, meta, raw = bundle
    meta[field] = value
    with pytest.raises(ValueError):
        validate_native_urdf_bundle(xml, json.dumps(meta).encode(), raw)


def test_changed_urdf_bytes_rejected(bundle: tuple) -> None:
    xml, meta, raw = bundle
    with pytest.raises(ValueError, match="URDF"):
        validate_native_urdf_bundle(xml + b" ", json.dumps(meta).encode(), raw)


def test_shared_contract_is_legacy_public_contract() -> None:
    from src.shared.python.motion_matching.native_urdf_contract import (
        validate_native_urdf_bundle as shared_contract,
    )

    assert shared_contract is validate_native_urdf_bundle
