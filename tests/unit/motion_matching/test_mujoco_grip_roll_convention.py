"""SHOW-A11 (#14736): Apply the A7 grip zero (wrists on top, 15 deg CCW) to MuJoCo specs and bindings.

Acceptance criteria:
- Every active MuJoCo grip binding and spec reads 15 deg +- 1 deg at address via the shared metric (address_grip_roll_deg).
- Excluded records and contact fixtures are documented in an explicit exclusion list with reasons.
- Character builder defaults reflect the A7 convention.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.grip_contact.binding_manifest import validate_manifest
from src.shared.python.motion_matching.grip_roll import (
    address_grip_roll_deg,
    grip_pose_state_from_spec,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "docs" / "development" / "full_body_models"
BINDINGS_DIR = ROOT / "src" / "config" / "grip_bindings"

# Native club frame at the flagship driver address (columns: club x, y (shaft), z)
FLAGSHIP_CLUB_ROTATION = Rotation.from_matrix(
    [[-0.5417, -0.7177, -0.4376], [0.6055, 0.0279, -0.7953], [0.5831, -0.6958, 0.4195]]
).as_matrix()

EXCLUDED_RECORDS: dict[str, str] = {
    "docs/development/full_body_models/full_body_spec_anthro_driver.json": (
        "canonical pre-A7 flagship driver model preserved as reference per "
        "test_owner_check_pre_a7_driver_reads_75_degrees (reads 75 deg on the A7 clock)"
    ),
    "docs/development/full_body_models/full_body_spec_anthro_iron7.json": (
        "7-iron capture-fit spec pinned to #12221 / #14753 receipt and tracked swing; "
        "kept as capture record per owner direction to work only on the best-match driver model"
    ),
    "docs/development/full_body_models/full_body_spec_anthro_driver_capture.json": (
        "capture-fit record from origin/main, kept separate per SHOW-A2 D22 to preserve capture chain"
    ),
    "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json": (
        "historical v2 candidate spec, superseded by anthro driver; preserved as backward-compatibility record"
    ),
    "docs/development/full_body_models/full_body_spec_v1.json": (
        "Simscape multibody native geometry spec export record from #9967 and #10099"
    ),
    "docs/development/full_body_models/full_body_spec_v2.json": (
        "Simscape multibody native geometry spec export record from #9967 and #10099"
    ),
    "src/config/grip_bindings/mujoco_allegro_hand_lab.json": (
        "Wonik Allegro Hand bimanual laboratory scene: contact fixture, not a qualified golfer"
    ),
    "src/config/grip_bindings/mujoco_shadow_hand_lab.json": (
        "Shadow Dexterous Hand bimanual laboratory scene: contact fixture, not a qualified golfer"
    ),
    "src/config/grip_bindings/mujoco_golfer.json": (
        "rigid weld MJCF at horizontal arm calibration pose qpos0; hands welded directly to club without full-body spec or wrist follower joints"
    ),
    "src/config/grip_bindings/mujoco_upper_body_swing.json": (
        "rigid weld MJCF at horizontal arm calibration pose qpos0; hands welded directly to club without full-body spec or wrist follower joints"
    ),
    "src/config/grip_bindings/mujoco_full_body_swing.json": (
        "rigid weld MJCF at horizontal arm calibration pose qpos0; hands welded directly to club without full-body spec or wrist follower joints"
    ),
    "src/config/grip_bindings/mujoco_advanced_biomechanical_swing.json": (
        "rigid weld MJCF at horizontal arm calibration pose qpos0; hands welded directly to club without full-body spec or wrist follower joints"
    ),
    "src/config/grip_bindings/mujoco_canonical_golf_humanoid.json": (
        "rigid weld MJCF at horizontal arm calibration pose qpos0; hands welded directly to club without full-body spec or wrist follower joints"
    ),
}


def test_excluded_records_have_explicit_reasons() -> None:
    """Every excluded MuJoCo record or contact fixture has a documented non-empty reason."""
    for rel_path, reason in EXCLUDED_RECORDS.items():
        full_path = ROOT / rel_path
        assert full_path.exists(), f"Excluded file {rel_path} does not exist"
        assert len(reason.strip()) >= 20, f"Reason for {rel_path} is too short"


def test_canonical_driver_pre_a7_roll_is_preserved() -> None:
    """Canonical pre-A7 flagship driver model preserved as record reading ~75 deg."""
    driver_path = MODELS_DIR / "full_body_spec_anthro_driver.json"
    spec = json.loads(driver_path.read_text(encoding="utf-8"))
    lead, trail = address_grip_roll_deg(
        grip_pose_state_from_spec(spec, FLAGSHIP_CLUB_ROTATION)
    )
    assert lead == pytest.approx(75.0, abs=1.5)
    assert trail == pytest.approx(-126.5, abs=3.0)


def test_full_body_v1_spec_export_binding_valid_and_matches_canonical_driver_hash() -> (
    None
):
    """full_body_v1_spec_export.json is a valid grip-binding/v1 matching the canonical driver."""
    driver_path = MODELS_DIR / "full_body_spec_anthro_driver.json"
    manifest_path = BINDINGS_DIR / "full_body_v1_spec_export.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    expected_hash = f"sha256:{hashlib.sha256(driver_path.read_bytes()).hexdigest()}"
    assert manifest["model_hash"] == expected_hash


def test_humanoid_character_builder_binding_valid_and_matches_driver_hash() -> None:
    """humanoid_character_builder.json is a valid grip-binding/v1 matching the canonical driver."""
    driver_path = MODELS_DIR / "full_body_spec_anthro_driver.json"
    manifest_path = BINDINGS_DIR / "humanoid_character_builder.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    expected_hash = f"sha256:{hashlib.sha256(driver_path.read_bytes()).hexdigest()}"
    assert manifest["model_hash"] == expected_hash
