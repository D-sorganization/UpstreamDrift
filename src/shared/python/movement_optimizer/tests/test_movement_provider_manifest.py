# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Regression tests for the shared Movement Optimizer provider manifest."""

from __future__ import annotations

from scripts.movement_optimizer_provider_manifest import (
    MOVEMENT_OPTIMIZER_PROVIDER_MANIFEST,
    REPO_ROOT,
    validate_movement_optimizer_provider_manifest,
)


def test_movement_provider_manifest_validates_against_repo_layout() -> None:
    """The published optimizer pack should resolve cleanly from the repo root."""
    manifest = validate_movement_optimizer_provider_manifest()

    assert manifest["pack_id"] == "tools-movement-optimizer"
    assert manifest["provider"] == "tools"
    assert len(manifest["models"]) == 1


def test_movement_provider_manifest_declares_shared_launcher_metadata() -> None:
    """The optimizer pack should expose launcher metadata for shared consumers."""
    manifest = validate_movement_optimizer_provider_manifest()
    entry = manifest["models"][0]

    assert entry["id"] == "tools_movement_optimizer"
    assert entry["capabilities"] == [
        "optimization",
        "biomechanics",
        "trajectory",
        "cli",
        "pyqt6",
        "swingset",
        "chain_dynamics",
    ]
    assert entry["launcher"]["category"] == "tool"
    assert entry["launcher"]["status"] == "provider_ready"
    assert entry["launcher"]["web_route"] == "/tools/movement-optimizer"


def test_movement_provider_manifest_points_at_console_entry_module() -> None:
    """The provider path should track the installed console entry module."""
    manifest = validate_movement_optimizer_provider_manifest()
    entry = manifest["models"][0]

    assert entry["source_root"] == "src/movement_optimizer"
    assert entry["path"] == "launch_pyqt6.py"


def test_movement_provider_manifest_stays_in_expected_location() -> None:
    """The shared optimizer manifest should remain a top-level provider artifact."""
    assert MOVEMENT_OPTIMIZER_PROVIDER_MANIFEST == (
        REPO_ROOT / "src" / "movement_optimizer" / "model_pack.yaml"
    )
    assert MOVEMENT_OPTIMIZER_PROVIDER_MANIFEST.is_file()
