"""Unit tests for named grip presets (HRG-GS-G5, #13303).

Pins the panel-consensus starting values for Palmer overlap, Woods interlock,
and ten-finger grips; verifies GR1 angle range satisfaction, photo_estimate
provenance recording (never measured), and loading across GR4 surfaces (CLI,
desktop, web).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.api.routes.character_appearance import (
    GripRevisionRequest,
    build_grip_revision,
)
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

    # Qualitative source & evidence still citation
    assert len(preset.qualitative_source) > 20
    assert len(preset.evidence_still_description) > 20


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

    # Qualitative source & evidence still citation
    assert len(preset.qualitative_source) > 20
    assert len(preset.evidence_still_description) > 20


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

    # Qualitative source & evidence still citation
    assert len(preset.qualitative_source) > 20
    assert len(preset.evidence_still_description) > 20


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
