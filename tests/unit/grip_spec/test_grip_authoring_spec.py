"""HRG-GB1 versioned grip authoring spec (#15123)."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
import numpy as np
import pytest

from dataclasses import replace

from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec import (
    GripForceProvider,
    GripSpec,
    HandGripSpec,
    HandSize,
    register_grip_force_provider,
    resolve_contact_parameters,
    resolve_grip_force_limits,
)
from src.shared.python.grip_spec.validation import validate_grip_style_fields
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)

pytestmark = pytest.mark.unit

_SHELLS_13997 = frozenset({"dorsum"})
_GOLDEN_MESH_DIGEST = "a197809270636c398441505e402274b6107ec032588d22f3563f173f2523393a"


def _mesh_summary_digest(pose: GripPose | None = None) -> str:
    pose = DEFAULT_GRIP_POSE if pose is None else pose
    radius = 0.014
    summary: dict[str, object] = {}
    for style in (GripStyle.OVERLAP,):
        for hand in Hand:
            result = build_grip_hand_mesh(hand, grip_radius_m=radius, pose=pose)
            hand_key = hand.value
            summary.setdefault(style.value, {})[hand_key] = {}
            for comp in result.components:
                if comp.name in _SHELLS_13997:
                    continue
                vertices = comp.mesh.vertices
                summary[style.value][hand_key][comp.name] = {
                    "on_grip": comp.on_grip,
                    "n_vertices": len(vertices),
                    "mean": np.round(vertices.mean(axis=0), 6).tolist(),
                }
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_legacy_spec_projects_to_default_grip_pose_exactly() -> None:
    projected = GripSpec.legacy().to_grip_pose()
    assert projected == DEFAULT_GRIP_POSE


def test_neutral_spec_thumb_azimuths_lead_22_5_trail_45() -> None:
    spec = GripSpec.neutral()
    assert spec.lead.thumb_azimuth_deg == pytest.approx(22.5)
    assert spec.trail.thumb_azimuth_deg == pytest.approx(45.0)


def test_thumb_azimuth_sign_mirrors_for_left_handed() -> None:
    right = GripSpec.neutral(handedness="right")
    left = GripSpec.neutral(handedness="left")
    assert right.resolved_thumb_azimuth_deg("lead") == pytest.approx(22.5)
    assert right.resolved_thumb_azimuth_deg("trail") == pytest.approx(45.0)
    assert left.resolved_thumb_azimuth_deg("lead") == pytest.approx(-22.5)
    assert left.resolved_thumb_azimuth_deg("trail") == pytest.approx(-45.0)


def test_spec_roundtrip_json_and_hash_stable() -> None:
    spec = GripSpec.neutral()
    blob = spec.to_json()
    restored = GripSpec.from_json(blob)
    assert restored == spec
    assert restored.spec_sha256 == spec.spec_sha256
    again = GripSpec.from_dict(json.loads(blob))
    assert again.spec_sha256 == spec.spec_sha256


def test_spec_rejects_out_of_range_fields() -> None:
    with pytest.raises(ValueError, match="club_cross_angle_deg"):
        GripSpec.legacy(lead=HandGripSpec(club_cross_angle_deg=120.0))
    with pytest.raises(ValueError, match="palm_finger_depth"):
        GripSpec.legacy(lead=HandGripSpec(palm_finger_depth=1.5))
    with pytest.raises(ValueError, match="length_m"):
        GripSpec.legacy(hand_size=HandSize(length_m=float("nan"), breadth_m=0.08))
    with pytest.raises(ValueError, match="schema_version"):
        GripSpec.from_dict({"schema_version": 99, "handedness": "right"})
    with pytest.raises(ValueError, match="thumb_azimuth_deg"):
        HandGripSpec(thumb_azimuth_deg=-5.0)
    with pytest.raises(ValueError, match="trail V offset"):
        GripSpec.legacy(
            lead=HandGripSpec(v_azimuth_deg=90.0),
            trail=HandGripSpec(v_azimuth_deg=-40.0),
        )


def test_left_handed_neutral_projects_unsigned_lead_thumb_offset() -> None:
    pose = GripSpec.neutral(handedness="left").to_grip_pose()
    assert pose.lead_thumb_offset_deg == pytest.approx(22.5)
    assert pose.handedness == "left"


def test_existing_grip_pose_and_grip_hand_mesh_outputs_unchanged() -> None:
    legacy_pose = GripSpec.legacy().to_grip_pose()
    assert legacy_pose == DEFAULT_GRIP_POSE
    assert _mesh_summary_digest(legacy_pose) == _GOLDEN_MESH_DIGEST
    assert _mesh_summary_digest() == _GOLDEN_MESH_DIGEST


def test_force_and_contact_params_resolved_from_provider() -> None:
    spec = GripSpec.legacy()
    limits = resolve_grip_force_limits(spec)
    assert limits["lead"] == "unlimited"
    assert limits["trail"] == "unlimited"
    contact = resolve_contact_parameters(spec)
    assert contact.provider_name == spec.contact_provider_name
    assert math.isfinite(contact.preload_penetration_m())


def test_grip_spec_public_constructors_validate() -> None:
    with pytest.raises(TypeError):
        GripSpec.legacy(style="overlap")  # type: ignore[arg-type]


def test_validate_grip_style_fields_rejects_non_style() -> None:
    with pytest.raises(TypeError, match="GripStyle"):
        validate_grip_style_fields("overlap", False)
    with pytest.raises(TypeError, match="bool"):
        validate_grip_style_fields(GripStyle.OVERLAP, 0)


# --- registration smoke (swappable by name) ---


@pytest.fixture
def _restore_force_provider_registry() -> None:
    from src.shared.python.grip_spec import providers as grip_providers

    saved = dict(grip_providers._FORCE_REGISTRY)
    try:
        yield
    finally:
        grip_providers._FORCE_REGISTRY.clear()
        grip_providers._FORCE_REGISTRY.update(saved)


def test_custom_force_provider_registers_and_resolves(
    _restore_force_provider_registry: None,
) -> None:
    provider_name = "unit-test-cap-grip-force-provider"

    @dataclasses.dataclass(frozen=True)
    class _CapProvider(GripForceProvider):
        cap_n: float

        def lead_force_limit(self) -> float | str:
            return self.cap_n

        def trail_force_limit(self) -> float | str:
            return "unlimited"

    register_grip_force_provider(provider_name, _CapProvider(cap_n=250.0))
    try:
        spec = replace(GripSpec.legacy(), force_provider_name=provider_name)
        limits = resolve_grip_force_limits(spec)
        assert limits["lead"] == pytest.approx(250.0)
        pose = spec.to_grip_pose()
        assert pose.lead_grip_force_limit_n == pytest.approx(250.0)
    finally:
        from src.shared.python.grip_spec import providers as grip_providers

        grip_providers._FORCE_REGISTRY.pop(provider_name, None)
