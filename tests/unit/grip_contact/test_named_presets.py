"""Unit tests for named grip presets (HRG-GS-G5, #13303).

Pins the panel-consensus starting values for Palmer overlap, Woods interlock,
and ten-finger grips; verifies GR1 angle range satisfaction, photo_estimate
provenance recording (never measured), and loading across GR4 surfaces (CLI,
desktop, web).
"""

from __future__ import annotations

import json
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.api.routes.character_appearance import (
    GripRevisionRequest,
    build_grip_revision,
)
from src.api.services.character_appearance_service import appearance_library
from src.shared.python.grip_contact.contact_topology import (
    GripStyle,
    contact_topology,
)
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.motion_matching.execution.grip_rotation import (
    GRIP_PARAMETERS,
    GRIP_PROVENANCE_SOURCES,
    apply_grip_config,
    grip_spec_of,
)
from src.shared.python.motion_matching.execution.grip_selection import (
    GRIP_PRESET_CHOICES,
    GripOverrides,
    apply_grip_selection,
    resolve_grip,
)
from src.shared.python.motion_matching.grip_contract import (
    NAMED_GRIP_PRESETS,
    GripSpec,
    NamedGripPreset,
    named_grip_preset,
)
from src.tools.character_builder.core import CharacterBuilderModel

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DRIVER = json.loads(
    (
        ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_text(encoding="utf-8")
)


def test_photo_estimate_added_to_grip_provenance_sources() -> None:
    """photo_estimate is an accepted source in GRIP_PROVENANCE_SOURCES."""
    assert "photo_estimate" in GRIP_PROVENANCE_SOURCES
    assert "measured" not in GRIP_PROVENANCE_SOURCES
    assert GRIP_PROVENANCE_SOURCES == (
        "user_declared",
        "rom_set_identified",
        "default",
        "photo_estimate",
    )


def test_palmer_overlap_pinned_values_and_ranges() -> None:
    """Palmer overlap pins lead/trail angles, spacing, trigger, style, and sources."""
    preset = named_grip_preset("palmer_overlap")
    assert isinstance(preset, NamedGripPreset)
    assert preset.name == "palmer_overlap"
    assert preset.style is GripStyle.OVERLAP
    assert preset.trigger_finger is True
    assert preset.hand_spacing_m == pytest.approx(0.074, abs=1e-6)

    # Lead hand angles
    assert preset.lead.alpha_deg == pytest.approx(12.0, abs=1e-6)
    assert preset.lead.psi_deg == pytest.approx(45.0, abs=1e-6)
    assert preset.lead.gamma_deg == pytest.approx(0.0, abs=1e-6)

    # Trail hand angles
    assert preset.trail.alpha_deg == pytest.approx(12.0, abs=1e-6)
    assert preset.trail.psi_deg == pytest.approx(35.0, abs=1e-6)
    assert preset.trail.gamma_deg == pytest.approx(0.0, abs=1e-6)

    # Angles are inside GR1 ranges
    spec = preset.spec
    assert -60.0 <= spec.lead.alpha_deg <= 60.0
    assert 5.0 <= spec.lead.psi_deg <= 85.0
    assert -180.0 <= spec.lead.gamma_deg <= 180.0
    assert -60.0 <= spec.trail.alpha_deg <= 60.0
    assert 5.0 <= spec.trail.psi_deg <= 85.0
    assert -180.0 <= spec.trail.gamma_deg <= 180.0
    assert 0.0 <= spec.pose().lead_v_azimuth_deg <= 90.0

    # Qualitative source & evidence still requirement
    assert len(preset.qualitative_source) > 20
    assert len(preset.evidence_still_requirement) > 20
    assert "American Hero" in preset.qualitative_source
    assert "golf.com" in preset.qualitative_source
    assert "Go for Broke" not in preset.qualitative_source
    assert "Golf Digest" not in preset.qualitative_source
    assert "Cherry Hills" not in preset.evidence_still_requirement
    assert "photo_estimate" in preset.evidence_still_requirement
    assert "no image is committed" in preset.evidence_still_requirement


def test_woods_interlock_pinned_values_and_ranges() -> None:
    """Woods interlock pins lead/trail angles, spacing, trigger, style, and sources."""
    preset = named_grip_preset("woods_interlock")
    assert isinstance(preset, NamedGripPreset)
    assert preset.name == "woods_interlock"
    assert preset.style is GripStyle.INTERLOCK
    assert preset.trigger_finger is False
    assert preset.hand_spacing_m == pytest.approx(0.069, abs=1e-6)

    # Lead hand angles
    assert preset.lead.alpha_deg == pytest.approx(2.0, abs=1e-6)
    assert preset.lead.psi_deg == pytest.approx(38.0, abs=1e-6)
    assert preset.lead.gamma_deg == pytest.approx(0.0, abs=1e-6)

    # Trail hand angles
    assert preset.trail.alpha_deg == pytest.approx(2.0, abs=1e-6)
    assert preset.trail.psi_deg == pytest.approx(34.0, abs=1e-6)
    assert preset.trail.gamma_deg == pytest.approx(0.0, abs=1e-6)

    # Angles are inside GR1 ranges
    spec = preset.spec
    assert -60.0 <= spec.lead.alpha_deg <= 60.0
    assert 5.0 <= spec.lead.psi_deg <= 85.0
    assert -180.0 <= spec.lead.gamma_deg <= 180.0
    assert -60.0 <= spec.trail.alpha_deg <= 60.0
    assert 5.0 <= spec.trail.psi_deg <= 85.0
    assert -180.0 <= spec.trail.gamma_deg <= 180.0
    assert 0.0 <= spec.pose().lead_v_azimuth_deg <= 90.0

    # Qualitative source & evidence still requirement
    assert len(preset.qualitative_source) > 20
    assert len(preset.evidence_still_requirement) > 20
    assert "Tiger Woods" in preset.qualitative_source
    assert "How I Play Golf" not in preset.qualitative_source
    assert "PGA Tour player profile" not in preset.qualitative_source
    assert (
        "2000-2001 major championship broadcast"
        not in preset.evidence_still_requirement
    )
    assert "photo_estimate" in preset.evidence_still_requirement
    assert "no image is committed" in preset.evidence_still_requirement


def test_ten_finger_pinned_values_and_ranges() -> None:
    """Ten-finger pins lead/trail angles, spacing, trigger, style, and sources."""
    preset = named_grip_preset("ten_finger")
    assert isinstance(preset, NamedGripPreset)
    assert preset.name == "ten_finger"
    assert preset.style is GripStyle.TEN_FINGER
    assert preset.trigger_finger is False
    assert preset.hand_spacing_m == pytest.approx(0.090, abs=1e-6)

    # Lead hand angles
    assert preset.lead.alpha_deg == pytest.approx(5.0, abs=1e-6)
    assert preset.lead.psi_deg == pytest.approx(40.0, abs=1e-6)
    assert preset.lead.gamma_deg == pytest.approx(0.0, abs=1e-6)

    # Trail hand angles
    assert preset.trail.alpha_deg == pytest.approx(5.0, abs=1e-6)
    assert preset.trail.psi_deg == pytest.approx(32.0, abs=1e-6)
    assert preset.trail.gamma_deg == pytest.approx(0.0, abs=1e-6)

    # Angles are inside GR1 ranges
    spec = preset.spec
    assert -60.0 <= spec.lead.alpha_deg <= 60.0
    assert 5.0 <= spec.lead.psi_deg <= 85.0
    assert -180.0 <= spec.lead.gamma_deg <= 180.0
    assert -60.0 <= spec.trail.alpha_deg <= 60.0
    assert 5.0 <= spec.trail.psi_deg <= 85.0
    assert -180.0 <= spec.trail.gamma_deg <= 180.0
    assert 0.0 <= spec.pose().lead_v_azimuth_deg <= 90.0

    # Qualitative source & evidence still requirement (no named person)
    assert len(preset.qualitative_source) > 20
    assert len(preset.evidence_still_requirement) > 20
    assert "Bob Rosburg" not in preset.qualitative_source
    assert "Rosburg" not in preset.qualitative_source
    assert "photo_estimate" in preset.evidence_still_requirement
    assert "no image is committed" in preset.evidence_still_requirement


@pytest.mark.parametrize("name", ["palmer_overlap", "woods_interlock", "ten_finger"])
def test_named_preset_roundtrip_with_photo_estimate_provenance(name: str) -> None:
    """Named presets round-trip through apply_grip_config with photo_estimate."""
    preset = named_grip_preset(name)
    spec = preset.spec
    revised = apply_grip_config(DRIVER, spec, provenance="photo_estimate")

    # Document has recorded grip_provenance
    prov = revised["subject"]["grip_provenance"]
    for param in GRIP_PARAMETERS:
        assert prov[param] == "photo_estimate"
        assert prov[param] != "measured"

    # Configuration matches
    assert grip_spec_of(revised) == spec

    # Idempotent apply
    reapplied = apply_grip_config(revised, spec, provenance="photo_estimate")
    assert reapplied == revised

    # Contact topology builds cleanly for this preset's spec
    topo = contact_topology(spec)
    assert topo.style is spec.style
    assert topo.trigger_finger is spec.trigger_finger


@pytest.mark.parametrize("name", ["palmer_overlap", "woods_interlock", "ten_finger"])
def test_named_presets_load_through_gr4_surfaces(name: str) -> None:
    """Named presets load through CLI overrides, Desktop model, and API request."""
    # 1. Resolver & CLI overrides
    overrides = GripOverrides(preset=name)
    assert overrides.has_physical is True
    assert name in GRIP_PRESET_CHOICES
    resolved = resolve_grip(overrides)
    expected_spec = named_grip_preset(name).spec
    assert resolved == expected_spec

    doc, sel_grip = apply_grip_selection(DRIVER, overrides)
    assert sel_grip == expected_spec
    for param in GRIP_PARAMETERS:
        assert doc["subject"]["grip_provenance"][param] == "photo_estimate"
        assert doc["subject"]["grip_provenance"][param] != "measured"

    # 2. Desktop model
    model = CharacterBuilderModel()
    model.set_grip("preset", name)
    assert model.resolved_grip() == expected_spec
    compiled = model.compile()
    assert compiled.document["subject"]["grip_config"]["hand_spacing_m"] == (
        expected_spec.hand_spacing_m
    )
    for param in GRIP_PARAMETERS:
        assert (
            compiled.document["subject"]["grip_provenance"][param] == "photo_estimate"
        )

    # 3. Web API request
    req = GripRevisionRequest(preset=name)  # type: ignore[arg-type]
    res = build_grip_revision(req)
    assert res["refit_required"] is True
    assert res["grip"]["hand_spacing_m"] == expected_spec.hand_spacing_m


def test_dbc_preconditions_reject_bad_inputs() -> None:
    """DbC preconditions reject unknown names and bad types."""
    with pytest.raises(ValueError, match="unknown named grip preset"):
        named_grip_preset("nonexistent")
    with pytest.raises(ValueError, match="unknown named grip preset"):
        GripSpec.named_preset("nonexistent")
    with pytest.raises(TypeError):
        named_grip_preset(123)  # type: ignore[arg-type]


def test_dbc_blank_string_is_value_error_and_wrong_type_is_type_error() -> None:
    """DbC: blank or empty string is ValueError, wrong type is TypeError."""
    base_spec = named_grip_preset("palmer_overlap").spec

    # Wrong types raise TypeError
    with pytest.raises(TypeError, match="name must be a str"):
        NamedGripPreset(
            name=123,  # type: ignore[arg-type]
            label="Label",
            spec=base_spec,
            qualitative_source="Source text",
            evidence_still_requirement="Still requirement",
        )
    with pytest.raises(TypeError, match="label must be a str"):
        NamedGripPreset(
            name="custom",
            label=123,  # type: ignore[arg-type]
            spec=base_spec,
            qualitative_source="Source text",
            evidence_still_requirement="Still requirement",
        )
    with pytest.raises(TypeError, match="spec must be a GripSpec"):
        NamedGripPreset(
            name="custom",
            label="Label",
            spec="not_a_spec",  # type: ignore[arg-type]
            qualitative_source="Source text",
            evidence_still_requirement="Still requirement",
        )
    with pytest.raises(TypeError, match="qualitative_source must be a str"):
        NamedGripPreset(
            name="custom",
            label="Label",
            spec=base_spec,
            qualitative_source=None,  # type: ignore[arg-type]
            evidence_still_requirement="Still requirement",
        )
    with pytest.raises(TypeError, match="evidence_still_requirement must be a str"):
        NamedGripPreset(
            name="custom",
            label="Label",
            spec=base_spec,
            qualitative_source="Source text",
            evidence_still_requirement=456,  # type: ignore[arg-type]
        )

    # Empty or whitespace-only strings raise ValueError
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        NamedGripPreset(
            name="   ",
            label="Label",
            spec=base_spec,
            qualitative_source="Source text",
            evidence_still_requirement="Still requirement",
        )
    with pytest.raises(ValueError, match="label must be a non-empty string"):
        NamedGripPreset(
            name="custom",
            label="",
            spec=base_spec,
            qualitative_source="Source text",
            evidence_still_requirement="Still requirement",
        )
    with pytest.raises(
        ValueError, match="qualitative_source must be a non-empty string"
    ):
        NamedGripPreset(
            name="custom",
            label="Label",
            spec=base_spec,
            qualitative_source="   ",
            evidence_still_requirement="Still requirement",
        )
    with pytest.raises(
        ValueError, match="evidence_still_requirement must be a non-empty string"
    ):
        NamedGripPreset(
            name="custom",
            label="Label",
            spec=base_spec,
            qualitative_source="Source text",
            evidence_still_requirement="",
        )


def test_named_grip_presets_is_read_only_mapping_proxy() -> None:
    """NAMED_GRIP_PRESETS is read-only types.MappingProxyType."""
    assert isinstance(NAMED_GRIP_PRESETS, types.MappingProxyType)
    with pytest.raises(TypeError):
        NAMED_GRIP_PRESETS["new_preset"] = NAMED_GRIP_PRESETS["palmer_overlap"]  # type: ignore[index]


def test_grip_spec_preset_rejects_extra_arguments_with_named_preset() -> None:
    """GripSpec.preset raises ValueError when extra arguments accompany a named preset."""
    # Default palm="neutral" and no shared is accepted
    spec = GripSpec.preset("palmer_overlap")
    assert spec == named_grip_preset("palmer_overlap").spec

    # Handedness alone is accepted
    spec_l = GripSpec.preset("palmer_overlap", handedness="left")
    assert spec_l.handedness == "left"

    # Non-neutral palm raises ValueError
    with pytest.raises(ValueError, match="does not accept extra arguments"):
        GripSpec.preset("palmer_overlap", palm="finger")

    # Extra shared kwargs raise ValueError
    with pytest.raises(ValueError, match="does not accept extra arguments"):
        GripSpec.preset("palmer_overlap", style=GripStyle.INTERLOCK)

    with pytest.raises(ValueError, match="does not accept extra arguments"):
        GripSpec.preset("woods_interlock", hand_spacing_m=0.08)


def test_named_preset_to_library_dict_matches_appearance_library() -> None:
    """Named preset entries in appearance_library use to_library_dict."""
    lib = appearance_library()
    named_dict = lib["grip"]["named_presets"]

    for name, preset in NAMED_GRIP_PRESETS.items():
        entry = named_dict[name]
        assert entry == preset.to_library_dict()
        assert entry["label"] == preset.label
        assert entry["qualitative_source"] == preset.qualitative_source
        assert entry["evidence_still_requirement"] == preset.evidence_still_requirement
        assert entry["hand_spacing_m"] == preset.hand_spacing_m
        assert entry["lead"] == {
            "alpha_deg": preset.lead.alpha_deg,
            "psi_deg": preset.lead.psi_deg,
            "gamma_deg": preset.lead.gamma_deg,
        }
        assert entry["trail"] == {
            "alpha_deg": preset.trail.alpha_deg,
            "psi_deg": preset.trail.psi_deg,
            "gamma_deg": preset.trail.gamma_deg,
        }


def test_apply_grip_selection_sidecar_with_physical_key_raises() -> None:
    """apply_grip_selection raises ValueError when sidecar carries physical key."""
    with pytest.raises(ValueError, match="only visual grip values"):
        apply_grip_selection(
            DRIVER,
            GripOverrides(),
            sidecar_grip={"hand_spacing_m": 0.08},
        )
    with pytest.raises(ValueError, match="only visual grip values"):
        apply_grip_selection(
            DRIVER,
            GripOverrides(preset="palmer_overlap"),
            sidecar_grip={"lead_below_butt_m": 0.04},
        )


def test_apply_grip_selection_restores_sidecar_visual_picks_in_returned_grip() -> None:
    """apply_grip_selection returned grip includes sidecar and override visual picks."""
    configured_doc = apply_grip_config(DRIVER, GripSpec())
    # A visual-only override + sidecar leaves the document physics hash unchanged
    doc_vis, grip_vis = apply_grip_selection(
        configured_doc,
        GripOverrides(grip_strength_deg=12.0),
        sidecar_grip={"lead_thumb_offset_deg": 18.0},
    )
    assert grip_vis is not None
    assert grip_vis.style is GripStyle.OVERLAP
    assert grip_vis.lead_thumb_offset_deg == pytest.approx(18.0)
    assert physics_spec_sha256(doc_vis) == physics_spec_sha256(configured_doc)

    # A physical style override mints a new revision while preserving sidecar visual picks
    overrides = GripOverrides(grip_style="ten_finger")
    doc, grip = apply_grip_selection(
        configured_doc,
        overrides,
        sidecar_grip={"lead_thumb_offset_deg": 18.0},
    )
    assert grip is not None
    assert grip.style is GripStyle.TEN_FINGER
    assert grip.lead_thumb_offset_deg == pytest.approx(18.0)
    assert physics_spec_sha256(doc) != physics_spec_sha256(configured_doc)
