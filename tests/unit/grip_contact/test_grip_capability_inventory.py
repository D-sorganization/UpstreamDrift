"""Tests for cross-model grip capability inventory (HRG-GS-M0, issue #13322).

Enforces:
1. Every hand-bearing model id in src/config/engine_model_inventory.json and the
   tiles in src/config/models.yaml is covered or excluded.
2. All referenced asset paths and evidence test paths exist.
3. Strict DbC validation on the loader and dataclasses (negative test cases).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pytest
import yaml

from src.shared.python.grip_contact.capability_inventory import (
    DEFAULT_INVENTORY_PATH,
    ExcludedModel,
    GripCapabilityInventory,
    GripFeatureCapability,
    ModelGripCapability,
    load_grip_capability_inventory,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ENGINE_INVENTORY_PATH = ROOT / "src" / "config" / "engine_model_inventory.json"
MODELS_YAML_PATH = ROOT / "src" / "config" / "models.yaml"

HAND_BEARING_FAMILIES = frozenset(
    {"full_body_anthro", "legacy_golfer", "golf_humanoid", "GolfSwing3D_Kinetic"}
)
HAND_BEARING_TILE_TYPES = frozenset(
    {"custom_humanoid", "drake", "pinocchio", "opensim", "myosim", "matlab_suite"}
)
EXCLUDED_TILE_TYPES = frozenset({"putting_green", "putting_green_gui"})


def _is_hand_bearing_package(pkg: dict[str, Any]) -> bool:
    """Detection rule for engine_model_inventory.json packages.

    A package is hand-bearing if its model_family represents a full-body or
    upper-body humanoid golfer asset equipped with hands or hand contact sites.
    """
    return pkg.get("model_family") in HAND_BEARING_FAMILIES


def _classify_models_yaml_tile(tile: dict[str, Any]) -> str:
    """Detection rule for src/config/models.yaml tiles.

    Returns:
        "hand_bearing" if the tile represents a physics engine launcher or tool
        for a hand-bearing humanoid golfer model.
        "excluded" if the tile represents a handless pendulum or putting model.
        "non_model_tool" for diagnostic, media-processing, or general utilities.
    """
    tid = str(tile.get("id", ""))
    ttype = str(tile.get("type", ""))
    cat = str(tile.get("launcher", {}).get("category", ""))
    path = str(tile.get("path", ""))

    if (
        ttype in EXCLUDED_TILE_TYPES
        or "putting" in tid
        or "pendulum" in tid
        or "double_pendulum" in path
    ):
        return "excluded"

    if (
        cat == "physics_engine"
        or ttype in HAND_BEARING_TILE_TYPES
        or tid == "character_builder"
    ):
        return "hand_bearing"

    return "non_model_tool"


# ---------------------------------------------------------------------------
# Coverage tests against configuration authorities
# ---------------------------------------------------------------------------


def test_every_hand_bearing_model_in_engine_inventory_is_covered() -> None:
    """Every hand-bearing package in engine_model_inventory.json must be covered."""
    inv = load_grip_capability_inventory()
    with ENGINE_INVENTORY_PATH.open("r", encoding="utf-8") as f:
        emi_data = json.load(f)

    hand_bearing_pkgs = [
        pkg["id"]
        for pkg in emi_data.get("packages", [])
        if _is_hand_bearing_package(pkg)
    ]
    assert len(hand_bearing_pkgs) == 16, (
        f"Expected 16 packages, got {len(hand_bearing_pkgs)}"
    )

    for pkg_id in hand_bearing_pkgs:
        assert inv.covers_model(pkg_id), (
            f"Package {pkg_id} is not covered in grip inventory"
        )


def test_every_hand_bearing_and_excluded_tile_in_models_yaml_is_handled() -> None:
    """Every hand-bearing launcher tile in models.yaml must be covered, and pendulum/putting excluded."""
    inv = load_grip_capability_inventory()
    with MODELS_YAML_PATH.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    hand_bearing_tiles: list[str] = []
    excluded_tiles: list[str] = []

    for tile in yaml_data.get("models", []):
        classification = _classify_models_yaml_tile(tile)
        tid = tile["id"]
        if classification == "hand_bearing":
            hand_bearing_tiles.append(tid)
        elif classification == "excluded":
            excluded_tiles.append(tid)

    assert len(hand_bearing_tiles) > 0
    assert len(excluded_tiles) > 0

    for tid in hand_bearing_tiles:
        assert inv.covers_model(tid), (
            f"Hand-bearing tile '{tid}' must be covered in grip inventory"
        )

    for tid in excluded_tiles:
        assert inv.is_excluded(tid), (
            f"Tile '{tid}' must be in excluded list of grip inventory"
        )
        excluded = inv.get_excluded(tid)
        assert excluded is not None
        assert bool(excluded.reason.strip()), (
            f"Excluded tile '{tid}' must have a non-empty reason"
        )


# ---------------------------------------------------------------------------
# Contract & schema compliance
# ---------------------------------------------------------------------------


def test_inventory_schema_and_path_existence() -> None:
    """Every entry in grip_capability_inventory.json must satisfy contracts."""
    inv = load_grip_capability_inventory()
    assert inv.schema == "grip-capability-inventory/v1"
    assert len(inv.models) >= 16

    required_features = {
        "visual_style",
        "allocated_hand_hand",
        "physical_pad_grip",
        "physical_hand_hand",
    }

    for model_id, model in inv.models.items():
        assert model.id == model_id
        assert model.engine in {
            "mujoco",
            "drake",
            "pinocchio",
            "opensim",
            "myosuite",
            "simscape",
            "jaxsim",
            "humanoid_character_builder",
        }
        # asset path exists
        assert (ROOT / model.asset_path).is_file(), (
            f"Missing asset_path: {model.asset_path}"
        )
        assert model.fingers in {"none", "rigid_proxy", "articulated"}
        assert set(model.features.keys()) == required_features

        for fname, feat in model.features.items():
            assert feat.tracking_issue.startswith("#")
            assert feat.tracking_issue[1:].isdigit()
            if not feat.available:
                assert bool(feat.reason.strip()), (
                    f"Missing reason for unavailable {fname} on {model_id}"
                )
            else:
                assert feat.evidence_test is not None
                assert (ROOT / feat.evidence_test).is_file(), (
                    f"Evidence test file not found: {feat.evidence_test} for {fname} on {model_id}"
                )


def test_evidence_policy_only_spec_export_path_has_available_features() -> None:
    """Today only full-body-v1 spec export paths may have available=True (with evidence_test)."""
    inv = load_grip_capability_inventory()

    for model_id, model in inv.models.items():
        for fname, feat in model.features.items():
            if feat.available:
                assert feat.evidence_test is not None
                # Allowed only for full_body_spec export models on physical_pad_grip
                assert fname == "physical_pad_grip"
                assert (
                    "spec" in model.asset_path
                    or "driver" in model_id
                    or "anthro" in model.asset_path
                )


# ---------------------------------------------------------------------------
# Negative DbC validation cases
# ---------------------------------------------------------------------------


def test_negative_invalid_schema(tmp_path: Path) -> None:
    """Loader rejects incorrect schema string."""
    bad_data = {
        "schema": "wrong-schema/v9",
        "description": "test",
        "models": [],
        "excluded": [],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad_data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_grip_capability_inventory(path)


def test_negative_feature_empty_reason_when_unavailable() -> None:
    """Feature with available=False must have non-empty reason."""
    with pytest.raises(ValueError, match="non-empty reason"):
        GripFeatureCapability(
            available=False,
            reason="",
            tracking_issue="#13326",
        )


def test_negative_feature_invalid_tracking_issue() -> None:
    """Tracking issue must match ^#\\d+$."""
    with pytest.raises(ValueError, match="tracking_issue"):
        GripFeatureCapability(
            available=False,
            reason="not implemented",
            tracking_issue="13326",  # missing '#'
        )
    with pytest.raises(ValueError, match="tracking_issue"):
        GripFeatureCapability(
            available=False,
            reason="not implemented",
            tracking_issue="#abc",  # not digits
        )


def test_negative_feature_available_without_evidence_test() -> None:
    """available=True requires evidence_test."""
    with pytest.raises(ValueError, match="evidence_test"):
        GripFeatureCapability(
            available=True,
            reason="implemented",
            tracking_issue="#13326",
            evidence_test=None,
        )


def test_negative_model_nonexistent_asset_path(tmp_path: Path) -> None:
    """Model with nonexistent asset path must fail validation."""
    feat = GripFeatureCapability(
        available=False,
        reason="pending",
        tracking_issue="#13326",
    )
    features = {
        "visual_style": feat,
        "allocated_hand_hand": feat,
        "physical_pad_grip": feat,
        "physical_hand_hand": feat,
    }
    with pytest.raises((ValueError, FileNotFoundError), match="asset_path"):
        ModelGripCapability(
            id="test/model",
            engine="mujoco",
            asset_path="nonexistent/path/to/asset.xml",
            attachment_mode="weld",
            fingers="none",
            features=features,
            repo_root=ROOT,
        )


def test_negative_model_invalid_fingers() -> None:
    """Invalid fingers token must fail validation."""
    feat = GripFeatureCapability(
        available=False,
        reason="pending",
        tracking_issue="#13326",
    )
    features = {
        "visual_style": feat,
        "allocated_hand_hand": feat,
        "physical_pad_grip": feat,
        "physical_hand_hand": feat,
    }
    with pytest.raises(ValueError, match="fingers"):
        ModelGripCapability(
            id="test/model",
            engine="mujoco",
            asset_path="docs/development/full_body_models/full_body_spec_anthro_driver.json",
            attachment_mode="weld",
            fingers="articulated_fingers_invalid",
            features=features,
            repo_root=ROOT,
        )


def test_negative_model_missing_required_features() -> None:
    """Model missing required features must fail validation."""
    feat = GripFeatureCapability(
        available=False,
        reason="pending",
        tracking_issue="#13326",
    )
    # Only 2 features provided
    features = {
        "visual_style": feat,
        "allocated_hand_hand": feat,
    }
    with pytest.raises(ValueError, match="features"):
        ModelGripCapability(
            id="test/model",
            engine="mujoco",
            asset_path="docs/development/full_body_models/full_body_spec_anthro_driver.json",
            attachment_mode="weld",
            fingers="none",
            features=features,
            repo_root=ROOT,
        )


def test_negative_excluded_empty_reason() -> None:
    """ExcludedModel requires a non-empty reason."""
    with pytest.raises(ValueError, match="reason"):
        ExcludedModel(id="putting_green", reason="")
