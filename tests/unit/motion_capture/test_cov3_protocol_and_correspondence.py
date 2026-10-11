"""Tests for Capture-O comparison protocol and landmark correspondence (COV-3, #11271)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.tour_capture_contract import tracked_labels
from src.shared.python.motion_pipeline.sources.hmr2_adapter import SMPL_BODY_JOINTS
from src.shared.python.pose_estimation.mediapipe_estimator import MediaPipeEstimator
from src.shared.python.pose_estimation.openpose_dnn_estimator import (
    OpenPoseDnnEstimator,
)
from src.shared.python.pose_estimation.rtmpose_onnx_estimator import COCO17_KEYPOINTS

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "src" / "config"
DOCS_DIR = REPO_ROOT / "docs" / "development" / "capture-o-video"

PROFILE_JSON_PATH = CONFIG_DIR / "cov_comparison_profile.v1.json"
CORRESPONDENCE_JSON_PATH = CONFIG_DIR / "cov_landmark_correspondence.json"
PROTOCOL_MD_PATH = DOCS_DIR / "comparison-protocol.md"


@pytest.mark.unit
def test_comparison_profile_json_schema_and_integrity() -> None:
    """Validate schema, comparison levels, hyperparameters, and metric definitions."""
    assert PROFILE_JSON_PATH.is_file(), f"Missing {PROFILE_JSON_PATH}"
    data = json.loads(PROFILE_JSON_PATH.read_text(encoding="utf-8"))

    assert data["$schema"] == "cov-comparison-profile/1.0.0"
    assert data["version"] == "1.0.0"
    assert data["profile_id"] == "cov_comparison_profile_v1"
    assert data["governing_issue"] == 11271
    assert data["parent_epic"] == 11268

    levels = data.get("comparison_levels", {})
    for expected_lvl in ("L0", "L1", "L2", "L3"):
        assert expected_lvl in levels, f"Missing level {expected_lvl}"
        assert "name" in levels[expected_lvl]
        assert "prerequisites" in levels[expected_lvl]
        assert "allowed_claims" in levels[expected_lvl]

    params = data.get("hyperparameters", {})
    pairing_params = params.get("pairing", {})
    assert pairing_params.get("tau_pair") == 0.65
    assert pairing_params.get("min_abstention_margin") == 0.05
    assert pairing_params.get("candidate_swings_count") == 13

    cam_params = params.get("virtual_camera", {})
    assert cam_params.get("reference_backend") == "mediapipe"
    assert cam_params.get("held_out_from_own_evaluation") is True

    phase_bins = data.get("phase_bins", {})
    assert len(phase_bins) == 10
    for idx in range(1, 11):
        p_key = f"P{idx}"
        assert p_key in phase_bins, f"Missing phase bin {p_key}"
        bin_info = phase_bins[p_key]
        assert "label" in bin_info
        assert 0.0 <= bin_info["start_phase"] < bin_info["end_phase"] <= 1.0

    metrics = data.get("metrics", [])
    assert len(metrics) >= 15
    for metric_entry in metrics:
        assert "id" in metric_entry
        assert metric_entry["level"] in ("L1", "L2", "L3")
        assert "quantity" in metric_entry
        assert "metric" in metric_entry
        assert "unit" in metric_entry
        assert "aggregation" in metric_entry
        assert isinstance(metric_entry["trustworthy_bound"], (int, float))
        assert isinstance(metric_entry["indicative_bound"], (int, float))


@pytest.mark.unit
def test_landmark_correspondence_json_schema_and_integrity() -> None:
    """Validate structure and detector specifications of correspondence table."""
    assert CORRESPONDENCE_JSON_PATH.is_file(), f"Missing {CORRESPONDENCE_JSON_PATH}"
    data = json.loads(CORRESPONDENCE_JSON_PATH.read_text(encoding="utf-8"))

    assert data["$schema"] == "cov-landmark-correspondence/1.0.0"
    assert data["version"] == "1.0.0"
    assert data["governing_issue"] == 11271
    assert data["parent_epic"] == 11268

    layouts = data.get("detector_layouts", {})
    assert "mediapipe_33" in layouts
    assert "coco_17" in layouts
    assert "openpose_25" in layouts
    assert "smpl_22" in layouts

    derived = data.get("derived_joint_centres", [])
    assert len(derived) >= 15
    for item in derived:
        assert "joint_name" in item
        assert "source_markers" in item
        assert len(item["source_markers"]) > 0
        assert "correspondence_type" in item
        assert item["correspondence_type"] in (
            "joint_centre",
            "surface_marker",
            "proxy_centroid",
        )
        assert "detector_targets" in item


@pytest.mark.unit
def test_landmark_correspondence_marker_names_exist_in_tracked_labels() -> None:
    """Verify all referenced markers exist in the canonical 34-marker tracked_labels."""
    canonical_markers = set(tracked_labels())
    assert len(canonical_markers) == 34

    data = json.loads(CORRESPONDENCE_JSON_PATH.read_text(encoding="utf-8"))
    for item in data.get("derived_joint_centres", []):
        for marker in item["source_markers"]:
            assert marker in canonical_markers, (
                f"Derived joint {item['joint_name']} references unknown marker: {marker}"
            )

    for excl in data.get("excluded_mappings", []):
        marker_or_kp = excl["marker_or_keypoint"]
        if marker_or_kp in canonical_markers:
            continue
        # If not a marker, it should be an explicit candidate or recognized feature
        assert excl["status"] == "excluded"


@pytest.mark.unit
def test_landmark_correspondence_detector_targets_exist_in_vocabularies() -> None:
    """Verify all mapped detector keypoints exist in their respective vocabulary definitions."""
    mediapipe_kps = set(MediaPipeEstimator.LANDMARK_MAP.values())
    coco17_kps = set(COCO17_KEYPOINTS)
    openpose_kps = set(OpenPoseDnnEstimator.LANDMARK_MAP.values())
    smpl_joints = set(SMPL_BODY_JOINTS)

    data = json.loads(CORRESPONDENCE_JSON_PATH.read_text(encoding="utf-8"))
    for item in data.get("derived_joint_centres", []):
        targets: dict[str, str | None] = item["detector_targets"]

        mp_target = targets.get("mediapipe_33")
        if mp_target is not None:
            assert mp_target in mediapipe_kps, (
                f"Invalid MediaPipe keypoint {mp_target} for joint {item['joint_name']}"
            )

        coco_target = targets.get("coco_17")
        if coco_target is not None:
            assert coco_target in coco17_kps, (
                f"Invalid COCO-17 keypoint {coco_target} for joint {item['joint_name']}"
            )

        openpose_target = targets.get("openpose_25")
        if openpose_target is not None:
            assert openpose_target in openpose_kps, (
                f"Invalid OpenPose keypoint {openpose_target} for joint {item['joint_name']}"
            )

        smpl_target = targets.get("smpl_22")
        if smpl_target is not None:
            assert smpl_target in smpl_joints, (
                f"Invalid SMPL joint {smpl_target} for joint {item['joint_name']}"
            )


@pytest.mark.unit
def test_correspondence_exclusions_have_valid_rationales() -> None:
    """Verify exclusions are documented with clear physical or technical rationales."""
    data = json.loads(CORRESPONDENCE_JSON_PATH.read_text(encoding="utf-8"))
    exclusions = data.get("excluded_mappings", [])
    assert len(exclusions) >= 6

    excluded_keys = {item["marker_or_keypoint"] for item in exclusions}
    assert "HeadFront" in excluded_keys
    assert "Marker_3:3:1" in excluded_keys

    for item in exclusions:
        assert item["status"] == "excluded"
        assert "rationale" in item
        assert len(item["rationale"].strip()) > 10


@pytest.mark.unit
def test_protocol_documentation_file_exists_and_contains_governed_sections() -> None:
    """Verify markdown protocol file contains ratified levels and rationales for 8 decisions."""
    assert PROTOCOL_MD_PATH.is_file(), f"Missing {PROTOCOL_MD_PATH}"
    content = PROTOCOL_MD_PATH.read_text(encoding="utf-8")

    assert "# Capture-O Video Comparison Protocol (COV-3)" in content
    assert "Null Baseline Principle" in content
    for lvl in ("L0 Qualitative", "L1 Envelope", "L2 Paired 2D", "L3 Paired 3D"):
        assert lvl in content

    for decision_num in range(1, 9):
        assert f"Decision {decision_num}:" in content
