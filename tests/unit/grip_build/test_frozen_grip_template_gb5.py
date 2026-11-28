"""HRG-GB5 frozen grip template export and fail-closed loader (#15127)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import numpy as np

from src.shared.python.grip_build.frozen_template import (
    FROZEN_TEMPLATE_SCHEMA_ID,
    FROZEN_TEMPLATE_SCHEMA_VERSION,
    TEMPLATE_BUILDER_VERSION,
    FrozenGripTemplate,
    FrozenTemplateStampError,
    build_frozen_template,
    load_frozen_template,
    write_frozen_template,
)
from src.shared.python.grip_build.full_builder import build_full_grip
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec, HandSize
from src.shared.python.model_appearance.grip_hand_mesh import (
    DEFAULT_GRIP_POSE,
    Hand,
    build_grip_hand_mesh,
    components_from_frozen_template,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY = REPO_ROOT / "src" / "config" / "grip_capability_inventory.json"
AUDIT = REPO_ROOT / "docs" / "development" / "full_body_models" / "HANDS_AUDIT.md"

_NEW_SPECS: tuple[tuple[str, GripSpec], ...] = (
    ("neutral_right_overlap", GripSpec.neutral()),
    (
        "strong_left_interlock",
        replace(
            GripSpec.neutral(handedness="left"),
            style=GripStyle.INTERLOCK,
            lead=replace(HandGripSpec(), thumb_azimuth_deg=22.5),
            trail=replace(HandGripSpec(), thumb_azimuth_deg=45.0),
            hand_size=HandSize(length_m=0.19),
        ),
    ),
    (
        "weak_right_ten_finger",
        replace(
            GripSpec.neutral(),
            style=GripStyle.TEN_FINGER,
            hand_size=HandSize(length_m=0.16),
        ),
    ),
)


def test_template_roundtrip_and_stamp(tmp_path: Path) -> None:
    """A built grip exports as stamped data and reloads byte-identically."""
    for key, spec in _NEW_SPECS:
        template = build_frozen_template(spec)
        assert template.spec_sha256 == spec.spec_sha256, key
        assert template.builder_version == TEMPLATE_BUILDER_VERSION, key
        assert template.schema == FROZEN_TEMPLATE_SCHEMA_ID, key
        assert template.schema_version == FROZEN_TEMPLATE_SCHEMA_VERSION, key
        assert tuple(h.role for h in template.hands) == ("lead", "trail"), key
        for hand in template.hands:
            assert hand.capsules, key
            assert {c.owner for c in hand.capsules} >= {
                "index",
                "middle",
                "ring",
                "little",
                "thumb",
            }, key

        blob = template.to_json()
        reloaded = FrozenGripTemplate.from_json(blob)
        assert reloaded.canonical_json() == template.canonical_json(), key

        path = write_frozen_template(template, tmp_path / f"{key}.json")
        on_disk = load_frozen_template(
            path,
            expected_spec_sha256=spec.spec_sha256,
            expected_builder_version=TEMPLATE_BUILDER_VERSION,
        )
        assert on_disk.canonical_json() == template.canonical_json(), key


def test_template_export_is_reproducible() -> None:
    """The golden set regenerates bit-for-bit, so templates are not run-order dependent."""
    for key, spec in _NEW_SPECS:
        first = build_frozen_template(spec).canonical_json()
        second = build_frozen_template(spec).canonical_json()
        assert first == second, key


def test_template_capsules_agree_with_builder_contacts() -> None:
    """Frozen capsule data reproduces the builder's own thumb-club clearance."""
    from src.shared.python.grip_build.club_sdf import ClubSDF
    from src.shared.python.grip_build.registry import get_contact_distance_model
    from src.shared.python.grip_contact.grip_geometry import get_grip_preset

    spec = GripSpec.neutral()
    template = build_frozen_template(spec)
    model = get_contact_distance_model()
    club = ClubSDF(
        geometry=get_grip_preset(), axial_offset_m=spec.lead.index_contact_station
    )
    clearances = dict(template.contact_clearances_m)
    thumb = [c for c in template.hand("lead").capsules if c.owner == "thumb"]
    nearest = min(
        model.club_clearance(
            np.asarray(c.p0, dtype=float),
            np.asarray(c.p1, dtype=float),
            c.radius_m,
            club,
        )
        for c in thumb
    )
    assert nearest == pytest.approx(clearances["_lead_thumb_club_clearance"], abs=1e-12)


def test_loader_rejects_spec_hash_mismatch(tmp_path: Path) -> None:
    """A template captured from a different spec is refused, not silently drawn."""
    spec = GripSpec.neutral()
    other = replace(spec, lead=replace(spec.lead, palm_finger_depth=0.8))
    assert other.spec_sha256 != spec.spec_sha256

    path = write_frozen_template(build_frozen_template(spec), tmp_path / "stale.json")
    with pytest.raises(FrozenTemplateStampError, match="spec_sha256"):
        load_frozen_template(
            path,
            expected_spec_sha256=other.spec_sha256,
            expected_builder_version=TEMPLATE_BUILDER_VERSION,
        )


def test_loader_rejects_builder_version_mismatch(tmp_path: Path) -> None:
    """A template from an older builder is refused before any capsule data is returned."""
    spec = GripSpec.neutral()
    path = write_frozen_template(build_frozen_template(spec), tmp_path / "old.json")
    with pytest.raises(FrozenTemplateStampError, match="builder_version"):
        load_frozen_template(
            path,
            expected_spec_sha256=spec.spec_sha256,
            expected_builder_version="gb5-0.0.9",
        )


def test_loader_rejects_missing_file_and_bad_schema(tmp_path: Path) -> None:
    """Absent files and an unknown schema both fail closed."""
    with pytest.raises(FileNotFoundError):
        load_frozen_template(tmp_path / "nope.json")

    template = build_frozen_template(GripSpec.neutral())
    raw = json.loads(template.to_json())
    raw["schema"] = "grip-frozen-template/v0"
    skewed = tmp_path / "skewed.json"
    skewed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_frozen_template(skewed)


def test_mesh_flag_off_output_unchanged() -> None:
    """Default calls take the analytic path only, so existing renders do not move."""
    for hand, role in ((Hand.LEAD, "lead"), (Hand.TRAIL, "trail")):
        baseline = build_grip_hand_mesh(
            hand, grip_radius_m=0.015, pose=DEFAULT_GRIP_POSE
        )
        explicit_off = build_grip_hand_mesh(
            hand,
            grip_radius_m=0.015,
            pose=DEFAULT_GRIP_POSE,
            frozen_template=None,
        )
        assert (
            explicit_off.mesh.vertices.tobytes() == baseline.mesh.vertices.tobytes()
        ), role
        assert explicit_off.mesh.faces.tobytes() == baseline.mesh.faces.tobytes(), role


def test_mesh_flag_on_draws_from_template() -> None:
    """With a verified template the shells come from the frozen capsule data."""
    spec = GripSpec.neutral()
    template = build_frozen_template(spec)
    built = build_grip_hand_mesh(
        Hand.LEAD,
        grip_radius_m=0.015,
        pose=DEFAULT_GRIP_POSE,
        frozen_template=template,
    )
    names = [c.name for c in built.components]
    assert names == [
        f"lead_{c.owner}_{c.segment}" for c in template.hand("lead").capsules
    ]
    assert len(names) == len(set(names))

    direct = components_from_frozen_template(template, "lead")
    assert len(direct) == len(template.hand("lead").capsules)


def test_mesh_refuses_stale_builder_template() -> None:
    """Drawing fails closed on a template stamped by a different frozen builder."""
    template = replace(
        build_frozen_template(GripSpec.neutral()), builder_version="gb5-0.0.9"
    )
    with pytest.raises(FrozenTemplateStampError, match="builder_version"):
        components_from_frozen_template(template, "lead")
    with pytest.raises(FrozenTemplateStampError, match="builder_version"):
        build_grip_hand_mesh(
            Hand.LEAD,
            grip_radius_m=0.015,
            pose=DEFAULT_GRIP_POSE,
            frozen_template=template,
        )


def test_hands_audit_inventory_records_frozen_template() -> None:
    """The new capability is listed per engine as not consumed, and audit stays in sync."""
    models = json.loads(INVENTORY.read_text(encoding="utf-8"))["models"]
    assert models, "inventory must list models"
    per_engine = json.loads(INVENTORY.read_text(encoding="utf-8"))[
        "frozen_template_engines"
    ]
    engines = {model["engine"] for model in models}
    assert set(per_engine) == engines

    from src.shared.python.grip_contact.capability_inventory import (
        load_grip_capability_inventory,
    )

    inventory = load_grip_capability_inventory()
    for engine in engines:
        assert inventory.consumes_frozen_template(engine) is False, engine
    # Unknown engines fail closed.
    assert inventory.consumes_frozen_template("no_such_engine") is False

    audit_text = AUDIT.read_text(encoding="utf-8")
    assert "procedural frozen template (data)" in audit_text
    matrix = audit_text.split("## Per-Asset Hand Capability Matrix", maxsplit=1)[1]
    matrix = matrix.split("\n## ", maxsplit=1)[0]
    import re

    row_ids = re.findall(r"^\|\s*`([^`]+)`\s*\|", matrix, re.MULTILINE)
    assert set(row_ids) == {model["id"] for model in models}


def test_full_build_and_template_share_one_build_pass() -> None:
    """Passing a prebuilt result avoids a second solve and keeps the stamp."""
    spec = GripSpec.neutral()
    result = build_full_grip(spec)
    template = build_frozen_template(spec, result=result)
    assert template.spec_sha256 == result.receipt.spec_sha256
    assert template.status == result.status.value


def test_rejects_non_spec_input() -> None:
    """Precondition: only a GripSpec may be exported."""
    with pytest.raises(TypeError):
        build_frozen_template("not-a-spec")  # type: ignore[arg-type]
