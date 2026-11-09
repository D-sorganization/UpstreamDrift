"""Tests for SHOW-A10 evidence, provenance manifest, and summary deliverable (#14735 / #14906)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.tools.native_viewer_export.key_frames import COLORS, GREEN_TARGET

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
EVIDENCE_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/show_a10"
)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def test_show_a10_summary_structure() -> None:
    content = (EVIDENCE_DIR / "SUMMARY.md").read_text(encoding="utf-8")
    for token in (
        "SHOW-A10",
        "mujoco_driver",
        "mujoco_driver_candidate_a9",
        "REJECTED (G3)",
        "fa7a9b7075",
        "## Outputs (SHA-256)",
        "## Inputs (SHA-256)",
        "## Source Stills Consumed (SHA-256)",
        "## Angle & Measurement Attribution",
        "## Acceptance Comparison vs 11-04 Baseline (`fa7a9b7075`)",
        "## Visual Artifacts",
        "## Reproduction Command",
        "show_a8",
        "trunk_segments",
    ):
        assert token in content
    assert "<A9 bundle>" not in content
    assert "not measured from rendered frames" in content.lower()


def test_show_a10_provenance_manifest_attribution() -> None:
    m_path = EVIDENCE_DIR / "provenance_manifest.json"
    manifest = json.loads(m_path.read_text(encoding="utf-8"))
    frame_attr = manifest["visual_frame_attribution"]
    assert frame_attr["is_verified_a9_bundle_render"] is False
    assert frame_attr["key_frame_source_dir"].endswith("/evidence/show_a8")
    assert frame_attr["side_by_side_source_dir"].endswith("/evidence/trunk_segments")

    repo_root = EVIDENCE_DIR.parents[4]
    assert len(manifest["source_stills"]) == 14
    for rel_path, recorded_sha in manifest["source_stills"].items():
        assert _sha256(repo_root / rel_path) == recorded_sha

    assert len(manifest["output_artifacts"]) == 10
    for fname, meta in manifest["output_artifacts"].items():
        assert meta["sha256"] == _sha256(EVIDENCE_DIR / fname)
        assert meta["is_fresh_a9_render"] is False and len(meta["source_stills"]) >= 1

    overlay = manifest["angle_attribution"]["clock_overlay_values_deg"]
    assert overlay["lead"] == pytest.approx(15.0)
    assert overlay["trail"] == pytest.approx(-100.73)
    assert overlay["classification"] == "illustrative_target_and_modelling_choice"
    assert overlay["is_measured_from_rendered_frames"] is False

    measured = manifest["angle_attribution"]["rendered_frame_hand_measurements"]
    assert measured["status"] == "unavailable"
    assert measured["lead_roll_deg"] is None and measured["trail_roll_deg"] is None

    quant = manifest["quantitative_refit_metrics"]
    assert quant["receipt_model_id"] == "mujoco_driver_candidate_a9"
    assert quant["verdict"] == "REJECTED (G3)" and not quant["is_physically_accepted"]
    assert quant["candidate_metrics"]["marker_rms_mm"] == pytest.approx(55.3)
    assert quant["candidate_metrics"]["closure_residual_mm"] == pytest.approx(76.8)

    video = manifest["video_delivery_manifest"]
    assert video["in_repo_video_artifacts"] == []
    assert video["a9_bundle_video_render_status"] == "unavailable_in_repo"


def test_show_a10_synthetic_fixture_rejects_false_attribution(tmp_path: Path) -> None:
    """Generator and validator derive provenance from inputs and reject false claims."""
    script_path = EVIDENCE_DIR / "render_show_a10_evidence.py"
    spec = importlib.util.spec_from_file_location("render_show_a10", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    syn_a8 = tmp_path / "show_a8"
    syn_trunk = tmp_path / "trunk_segments"
    syn_a9 = tmp_path / "show_a9"
    syn_out = tmp_path / "show_a10"
    for d in (syn_a8, syn_trunk, syn_a9, syn_out):
        d.mkdir()

    for name in mod.FACE_ON_NAMES + mod.DTL_NAMES:
        Image.new("RGB", (64, 64), (10, 20, 30)).save(syn_a8 / name, format="PNG")
        Image.new("RGB", (64, 64), (70, 80, 90)).save(syn_out / name, format="PNG")

    for out_name, suffix in mod.SBS_SPECS:
        b_img = syn_trunk / f"before_v1_rigid_{suffix}.png"
        a_img = syn_trunk / f"after_v2_segmented_{suffix}.png"
        Image.new("RGB", (32, 32), (40, 50, 60)).save(b_img, format="PNG")
        Image.new("RGB", (32, 32), (45, 55, 65)).save(a_img, format="PNG")
        Image.new("RGB", (64, 64), (70, 80, 90)).save(syn_out / out_name, format="PNG")

    receipt_file = syn_a9 / "receipt.json"
    receipt_data = {
        "issue": 14734,
        "model_id": "synthetic_candidate_a9",
        "input_hashes": {"capture_sha256": "1" * 64, "baseline_commit": "abc1234567"},
        "address_grip": {
            "target_deg": 15.0,
            "lead_roll_deg": 15.0,
            "trail_roll_deg": -100.73,
        },
        "comparison_with_baseline": {
            "marker_rms_mm": {"baseline": 50.9, "candidate": 55.3},
            "closure_residual_mm": {"baseline": 62.2, "candidate": 76.8},
        },
        "acceptance_gates": {
            "horizon": "G3",
            "status": "REJECTED",
            "is_physically_accepted": False,
        },
    }
    receipt_file.write_text(json.dumps(receipt_data), encoding="utf-8")

    manifest = mod.build_provenance_manifest(
        syn_out, syn_a8, syn_trunk, receipt_file, tmp_path
    )
    quant = manifest["quantitative_refit_metrics"]
    assert quant["receipt_model_id"] == "synthetic_candidate_a9"
    assert manifest["receipt_inputs"]["show_a9_receipt_sha256"] == _sha256(receipt_file)

    bad_frame = json.loads(json.dumps(manifest))
    bad_frame["visual_frame_attribution"]["is_verified_a9_bundle_render"] = True
    with pytest.raises(ValueError, match="verified_a9_bundle_render"):
        mod.validate_provenance_manifest(bad_frame)

    bad_angle = json.loads(json.dumps(manifest))
    overlay = bad_angle["angle_attribution"]["clock_overlay_values_deg"]
    overlay["is_measured_from_rendered_frames"] = True
    with pytest.raises(ValueError, match="measured_from_rendered_frames"):
        mod.validate_provenance_manifest(bad_angle)

    bad_verdict = json.loads(json.dumps(manifest))
    bad_verdict["quantitative_refit_metrics"]["is_physically_accepted"] = True
    with pytest.raises(ValueError, match="physically_accepted"):
        mod.validate_provenance_manifest(bad_verdict)


def test_show_a10_output_pngs_valid_and_hashes_match() -> None:
    summary_content = (EVIDENCE_DIR / "SUMMARY.md").read_text(encoding="utf-8")

    expected_files = [
        "down_the_line_address_1280.png",
        "down_the_line_impact_1280.png",
        "down_the_line_top_1280.png",
        "face_on_address_1280.png",
        "face_on_impact_1280.png",
        "face_on_top_1280.png",
        "side_by_side_address_dtl.png",
        "side_by_side_address_face_on.png",
        "side_by_side_top_dtl.png",
        "side_by_side_top_face_on.png",
    ]

    for fname in expected_files:
        png_path = EVIDENCE_DIR / fname
        assert png_path.is_file(), f"Expected output {fname} not found"
        # Verify valid PNG header
        header = png_path.read_bytes()[:8]
        assert header == b"\x89PNG\r\n\x1a\n", f"{fname} is not a valid PNG"

        # Verify image opens and has dimensions
        img = Image.open(png_path)
        assert img.width > 0 and img.height > 0

        # Verify hash matches SUMMARY.md
        digest = _sha256(png_path)
        expected_entry = f"`{fname}` | `{digest}`"
        assert expected_entry in summary_content


def test_show_a10_face_on_has_grip_clock_inset() -> None:
    fo_addr = Image.open(EVIDENCE_DIR / "face_on_address_1280.png")
    arr = np.asarray(fo_addr)
    # Clock is at bottom right: center around w - 110, h - 120
    # Check green tick (60, 220, 60)
    has_green = np.any((arr[:, :, 1] > 200) & (arr[:, :, 0] < 100))
    assert has_green, "face_on_address should contain green target tick"

    # Check red lead hand (255, 60, 60)
    has_red = np.any((arr[:, :, 0] > 200) & (arr[:, :, 2] < 100))
    assert has_red, "face_on_address should contain red lead needle"

    # Check blue trail hand (80, 120, 255)
    has_blue = np.any((arr[:, :, 2] > 200) & (arr[:, :, 0] < 120))
    assert has_blue, "face_on_address should contain blue trail needle"


def test_show_a10_side_by_side_composition() -> None:
    sbs = Image.open(EVIDENCE_DIR / "side_by_side_address_face_on.png")
    # Base (720x720) + Candidate (720x720) = 1440x720
    assert sbs.size == (1440, 720)
