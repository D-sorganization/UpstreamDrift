"""One grip resolver for the CLI, desktop and web (HRG-GR4, #13205)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from src.api.services.character_appearance_service import (
    appearance_library,
    build_appearance,
    grip_revision_summary,
)
from src.shared.python.model_appearance.grip_pose import GripStyle
from src.shared.python.model_appearance.schema import (
    document_from_dict,
    physics_spec_sha256,
)
from src.shared.python.motion_matching.execution import spec_builder
from src.shared.python.motion_matching.execution.grip_rotation import (
    GRIP_PARAMETERS,
    apply_grip_config,
)
from src.shared.python.motion_matching.execution.grip_selection import (
    PHYSICAL_PICKS,
    VISUAL_KEYS,
    VISUAL_PICKS,
    GripOverrides,
    apply_grip_selection,
    apply_visual_picks,
    check_sidecar_grip,
    migrate_sidecar_grip,
    overrides_from_args,
    resolve_grip,
)
from src.shared.python.motion_matching.grip_contract import (
    GripSpec,
    grip_spec_to_physical,
)
from src.tools.character_builder.core import CharacterBuilderModel
from src.tools.native_viewer_export import cli

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = json.loads(
    (ROOT / "tests/fixtures/grip/grip_roundtrip.json").read_text(encoding="utf-8")
)
CASES = FIXTURE["cases"]
DRIVER = json.loads(
    (
        ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_text(encoding="utf-8")
)
_FLAGS = {
    "preset": "--grip-preset",
    "lead_shaft_rotation": "--lead-shaft-rotation",
    "trail_shaft_rotation": "--trail-shaft-rotation",
    "lead_palm_diagonal": "--lead-palm-diagonal",
    "trail_palm_diagonal": "--trail-palm-diagonal",
    "hand_spacing": "--hand-spacing",
    "grip_style": "--grip-style",
    "grip_thumb_offset_deg": "--grip-thumb-offset-deg",
    "grip_strength_deg": "--visual-grip-strength",
    "lead_roll_offset_deg": "--lead-roll-offset",
    "trail_roll_offset_deg": "--trail-roll-offset",
}
IDS = [c["name"] for c in CASES]


def _spec_builder_overrides(picks: dict[str, Any]) -> GripOverrides:
    """Overrides as ``spec_builder`` parses them from the equivalent flags."""
    parser = argparse.ArgumentParser()
    spec_builder.add_grip_arguments(parser, physical=True)
    argv: list[str] = []
    for key, flag in _FLAGS.items():
        if key in picks:
            argv += [flag, str(picks[key])]
    extra = {k: v for k, v in picks.items() if k not in _FLAGS}
    return GripOverrides.from_picks(
        {**vars(overrides_from_args(parser.parse_args(argv))), **extra}
    )


def _summary(document: dict[str, Any], grip: GripSpec) -> dict[str, Any]:
    return {
        "grip": grip_spec_to_physical(grip),
        "physics_spec_sha256": physics_spec_sha256(document),
    }


def test_grip_style_is_classified_as_physical_and_not_visual() -> None:
    # HRG-GS-G1 (#13346) classifies grip_style as physical while keeping
    # HRG-A2's (#13166) three visual keys alongside grip_thumb_offset_deg.
    assert "grip_style" in PHYSICAL_PICKS
    assert "grip_style" not in VISUAL_PICKS
    assert VISUAL_PICKS == (
        "grip_thumb_offset_deg",
        "grip_strength_deg",
        "lead_roll_offset_deg",
        "trail_roll_offset_deg",
    )
    assert VISUAL_KEYS == (
        "lead_thumb_offset_deg",
        "grip_strength_deg",
        "lead_roll_offset_deg",
        "trail_roll_offset_deg",
    )


@pytest.mark.parametrize("style", ["interlock", "ten_finger"])
def test_style_only_pick_gives_refit_required_and_new_physics_sha(style: str) -> None:
    parent_driver_sha = physics_spec_sha256(DRIVER)
    overrides = GripOverrides(grip_style=style)
    assert overrides.has_physical is True
    assert overrides.has_visual is False

    document, grip = apply_grip_selection(DRIVER, overrides)
    assert grip is not None
    assert grip.style is GripStyle(style)
    assert document["spec_revision"]["configuration"] == "grip"
    assert document["subject"]["grip_config"]["style"] == style
    assert document["subject"]["grip_provenance"] == dict.fromkeys(
        GRIP_PARAMETERS, "user_declared"
    )
    assert "grip configuration revision" in document["provenance"]
    assert physics_spec_sha256(document) != parent_driver_sha

    base_char = CharacterBuilderModel().compile()
    summary = grip_revision_summary(base_char, overrides)
    assert summary["refit_required"] is True
    assert summary["physics_spec_sha256"] != physics_spec_sha256(base_char.document)
    assert summary["spec_sha256"] != base_char.sha256
    assert summary["grip"]["style"] == style


def test_unchanged_default_style_pick_keeps_v1_bytes_identical() -> None:
    driver_bytes = json.dumps(DRIVER, sort_keys=True).encode("utf-8")
    driver_sha = physics_spec_sha256(DRIVER)
    document, _grip = apply_grip_selection(DRIVER, GripOverrides(grip_style="overlap"))
    assert json.dumps(document, sort_keys=True).encode("utf-8") == driver_bytes
    assert physics_spec_sha256(document) == driver_sha
    assert "spec_revision" not in document
    assert "grip_config" not in document["subject"]

    base_char = CharacterBuilderModel().compile()
    base_bytes = json.dumps(base_char.document, sort_keys=True).encode("utf-8")
    model = CharacterBuilderModel()
    model.set_grip("grip_style", "overlap")
    compiled = model.compile()
    assert json.dumps(compiled.document, sort_keys=True).encode("utf-8") == base_bytes
    assert compiled.sha256 == base_char.sha256
    summary = grip_revision_summary(base_char, GripOverrides(grip_style="overlap"))
    assert summary["refit_required"] is False
    assert summary["spec_sha256"] == base_char.sha256
    assert summary["physics_spec_sha256"] == physics_spec_sha256(base_char.document)


@pytest.mark.parametrize("physical_style", list(GripStyle))
@pytest.mark.parametrize("sidecar_style", list(GripStyle))
def test_sidecar_and_physical_style_can_never_disagree(
    physical_style: GripStyle, sidecar_style: GripStyle
) -> None:
    spec = GripSpec(style=physical_style)
    doc = apply_grip_config(DRIVER, spec)
    legacy_sidecar = {
        "style": sidecar_style.value,
        "lead_thumb_offset_deg": 18.0,
        "grip_strength_deg": 10.0,
        "lead_roll_offset_deg": -5.0,
        "trail_roll_offset_deg": 8.0,
    }

    if physical_style is not sidecar_style:
        pattern = rf"({physical_style.value}.*{sidecar_style.value}|{sidecar_style.value}.*{physical_style.value})"
        with pytest.raises(ValueError, match=pattern):
            migrate_sidecar_grip(legacy_sidecar, doc)
        with pytest.raises(ValueError, match=pattern):
            resolve_grip(None, spec, legacy_sidecar)
        with pytest.raises(ValueError, match=pattern):
            apply_grip_selection(doc, GripOverrides(), legacy_sidecar)
    else:
        migrated = migrate_sidecar_grip(legacy_sidecar, doc)
        assert migrated == {
            "lead_thumb_offset_deg": 18.0,
            "grip_strength_deg": 10.0,
            "lead_roll_offset_deg": -5.0,
            "trail_roll_offset_deg": 8.0,
        }
        assert "style" not in migrated
        resolved = resolve_grip(None, spec, legacy_sidecar)
        assert resolved.style is physical_style
        assert resolved.lead_thumb_offset_deg == pytest.approx(18.0)
        out_doc, out_grip = apply_grip_selection(doc, GripOverrides(), legacy_sidecar)
        assert out_grip is not None
        assert out_grip.style is physical_style
        assert out_grip.lead_thumb_offset_deg == pytest.approx(18.0)
        assert physics_spec_sha256(out_doc) == physics_spec_sha256(doc)


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_resolver_and_cli_match_the_fixture(case: dict[str, Any]) -> None:
    expected = case["expected"]
    for overrides in (
        GripOverrides.from_picks(case["picks"]),
        _spec_builder_overrides(case["picks"]),
    ):
        document, grip = apply_grip_selection(DRIVER, overrides)
        assert physics_spec_sha256(document) == expected["driver_physics_spec_sha256"]
        if expected["grip"] is not None:
            assert grip is not None
            assert grip_spec_to_physical(grip) == expected["grip"]
        assert overrides.has_physical == expected["refit_required"]
        if not expected["refit_required"]:  # picture-only: the parent's hash
            assert (
                expected["driver_physics_spec_sha256"]
                == (FIXTURE["parent"]["driver_physics_spec_sha256"])
            )


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_desktop_and_web_api_match_the_fixture(case: dict[str, Any]) -> None:
    expected = case["expected"]
    model = CharacterBuilderModel()
    for key, value in case["picks"].items():
        model.set_grip(key, value)
    compiled = model.compile()
    summary = grip_revision_summary(
        CharacterBuilderModel().compile(), GripOverrides.from_picks(case["picks"])
    )
    assert (
        physics_spec_sha256(compiled.document)
        == (expected["character_physics_spec_sha256"])
    )
    assert summary["physics_spec_sha256"] == expected["character_physics_spec_sha256"]
    assert summary["spec_sha256"] == compiled.sha256
    assert summary["refit_required"] == expected["refit_required"]
    if expected["grip"] is not None:
        assert summary["grip"] == expected["grip"]
    sidecar = model.appearance_document().get("grip", {})
    assert sidecar == expected["sidecar_grip"]
    visual = {k: v for k, v in case["picks"].items() if k in VISUAL_PICKS}
    assert build_appearance(visual, None).get("grip", {}) == sidecar


def test_visual_picks_never_move_the_physics_hash() -> None:
    parent = physics_spec_sha256(DRIVER)
    overrides = GripOverrides(grip_thumb_offset_deg=30.0)
    document, _grip = apply_grip_selection(DRIVER, overrides)
    assert physics_spec_sha256(document) == parent
    revised, _ = apply_grip_selection(DRIVER, GripOverrides(preset="strong"))
    re_thumbed, _ = apply_grip_selection(revised, overrides)
    assert physics_spec_sha256(re_thumbed) == physics_spec_sha256(revised)
    assert physics_spec_sha256(revised) != parent
    assert revised["spec_revision"]["configuration"] == "grip"


def test_a_physical_pick_is_refused_in_a_sidecar() -> None:
    with pytest.raises(ValueError, match=r"only visual grip values.*lead_v_azimuth"):
        check_sidecar_grip({"style": "overlap", "lead_v_azimuth_deg": 40.0})
    with pytest.raises(ValueError, match=r"only visual grip values.*style"):
        check_sidecar_grip({"style": "interlock"})
    with pytest.raises(ValueError, match="new model revision"):
        resolve_grip(None, GripSpec(), {"hand_spacing_m": 0.1})
    with pytest.raises(ValueError, match="picture-only"):
        apply_visual_picks({}, GripOverrides(preset="weak"))
    with pytest.raises(ValueError, match="picture-only"):
        apply_visual_picks({}, GripOverrides(grip_style="interlock"))


def test_native_viewer_refuses_a_physical_flag_and_takes_a_visual_one() -> None:
    base = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
    parse = cli.build_parser().parse_args
    with pytest.raises(ValueError, match="new model revision.*refit"):
        cli.build_settings(parse([*base, "--grip-preset", "strong"]))
    with pytest.raises(ValueError, match="new model revision.*refit"):
        cli.build_settings(parse([*base, "--grip-style", "ten_finger"]))
    settings = cli.build_settings(parse([*base, "--grip-thumb-offset-deg", "20"]))
    assert settings.appearance["grip"] == {"lead_thumb_offset_deg": 20.0}
    document_from_dict(settings.appearance)


def test_native_viewer_refuses_a_physical_value_in_a_sidecar(tmp_path: Path) -> None:
    sidecar = tmp_path / "x.appearance.json"
    sidecar.write_text(
        json.dumps({"schema_version": "appearance-v1", "grip": {"hand_spacing_m": 1}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only visual grip values"):
        cli.load_appearance_document(sidecar)


def test_precedence_pick_then_document_then_default() -> None:
    default = resolve_grip(None)
    assert default == GripSpec()
    doc = GripSpec.preset("strong", hand_spacing_m=0.09)
    assert resolve_grip(None, doc) == doc
    picked = resolve_grip(GripOverrides(preset="weak", hand_spacing=0.1), doc)
    assert picked.lead.alpha_deg == -15.0 and picked.hand_spacing_m == 0.1
    exact = resolve_grip(GripOverrides(preset="weak", lead_shaft_rotation=3.0), doc)
    assert exact.lead.alpha_deg == 3.0 and exact.trail.alpha_deg == -15.0
    restyled = resolve_grip(GripOverrides(grip_style="interlock"), doc)
    assert restyled.style is GripStyle.INTERLOCK
    sidecar = resolve_grip(
        None, doc, {"style": "overlap", "lead_thumb_offset_deg": 22.0}
    )
    assert sidecar.style is GripStyle.OVERLAP
    assert sidecar.lead_thumb_offset_deg == 22.0


def test_a_fit_outside_the_golf_ranges_needs_a_full_pick() -> None:
    with pytest.raises(ValueError, match="not a golf grip"):
        apply_grip_selection(DRIVER, GripOverrides(lead_palm_diagonal=45.0))
    document, grip = apply_grip_selection(
        DRIVER, GripOverrides(grip_thumb_offset_deg=20.0)
    )
    assert grip is None and document["subject"] == DRIVER["subject"]


def test_invalid_picks_fail_closed() -> None:
    with pytest.raises(ValueError, match="grip preset"):
        GripOverrides(preset="mighty")
    with pytest.raises(ValueError, match="grip style"):
        GripOverrides(grip_style="baseball")
    with pytest.raises(ValueError, match="unknown grip picks"):
        GripOverrides.from_picks({"spin": 1})
    with pytest.raises(ValueError):
        resolve_grip(GripOverrides(lead_palm_diagonal=200.0))
    with pytest.raises(ValueError, match="unknown grip pick"):
        CharacterBuilderModel().set_grip("spin", 1.0)
    model = CharacterBuilderModel()
    with pytest.raises(ValueError):
        model.set_grip("lead_palm_diagonal", 200.0)
    assert model.grip_picks() == {}


def test_library_lists_the_grip_presets_and_styles() -> None:
    grip = appearance_library()["grip"]
    assert grip["presets"] == {"weak": -15.0, "neutral": 0.0, "strong": 15.0}
    assert grip["styles"] == ["overlap", "interlock", "ten_finger"]


def _build(tmp_path: Path, *flags: str) -> tuple[Path, dict[str, Any]]:
    from src.shared.python.motion_matching.execution import assets

    out = tmp_path / "out"
    code = spec_builder.main(
        [
            "--native",
            str(assets.get_native_geometry_spec(None)),
            "--osim",
            str(assets.get_opensim_model(None)),
            "--native-candidate",
            str(assets.get_candidate_geometry_spec(None)),
            "--stature",
            "1.80",
            "--mass",
            "84",
            "--output",
            str(out),
            "--name",
            "grip_case",
            *flags,
        ]
    )
    assert code == 0
    spec = out / "grip_case.json"
    return spec, json.loads(spec.read_text(encoding="utf-8"))


def test_spec_builder_flags_write_a_revision_and_a_picture_sidecar(
    tmp_path: Path,
) -> None:
    parent_spec, parent = _build(tmp_path / "a")
    spec, doc = _build(
        tmp_path / "b",
        "--grip-preset",
        "weak",
        "--hand-spacing",
        "0.09",
        "--grip-style",
        "interlock",
        "--grip-thumb-offset-deg",
        "20",
    )
    assert doc["spec_revision"]["configuration"] == "grip"
    assert doc["subject"]["grip_config"]["lead"]["alpha_deg"] == -15.0
    assert doc["subject"]["grip_config"]["hand_spacing_m"] == 0.09
    assert doc["subject"]["grip_config"]["style"] == "interlock"
    assert physics_spec_sha256(doc) != physics_spec_sha256(parent)
    receipt = json.loads(
        (spec.parent / "build_receipt_grip_case.json").read_text(encoding="utf-8")
    )
    assert receipt["subject"]["grip_config"] == doc["subject"]["grip_config"]
    sidecar = json.loads(spec.with_name("grip_case.appearance.json").read_text())
    assert sidecar["grip"] == {"lead_thumb_offset_deg": 20.0}
    assert "style" not in sidecar["grip"]
    assert "grip_config" not in sidecar and "hand_spacing_m" not in sidecar["grip"]
    assert not parent_spec.with_name("grip_case.appearance.json").exists()


def test_spec_builder_style_only_writes_a_revision_without_a_sidecar(
    tmp_path: Path,
) -> None:
    parent_spec, parent = _build(tmp_path / "a")
    spec, doc = _build(tmp_path / "b", "--grip-style", "ten_finger")
    assert physics_spec_sha256(doc) != physics_spec_sha256(parent)
    assert doc["spec_revision"]["configuration"] == "grip"
    assert doc["subject"]["grip_config"]["style"] == "ten_finger"
    assert not spec.with_name("grip_case.appearance.json").exists()

    same_spec, same_doc = _build(tmp_path / "c", "--grip-style", "overlap")
    assert physics_spec_sha256(same_doc) == physics_spec_sha256(parent)
    assert "spec_revision" not in same_doc
    assert same_spec.read_bytes() == parent_spec.read_bytes()
