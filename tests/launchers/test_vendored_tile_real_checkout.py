"""Tiles resolve against the real vendor/ud-tools checkout (#12993).

Consumer integration test: it needs the materialised Tools tree, so it lives
under tests/launchers/ (run against the real Tools tree), not unit-test-gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.config.model_registry import ModelRegistry
from src.shared.python.config.model_source_providers import resolve_model_artifact
from src.shared.python.config.tile_target_resolution import resolve_tile_target
from src.shared.python.config.tools_vendor_authority import (
    inspect_tools_vendor_authority,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TILES = ("pendulum_simulator", "swing_objective_lab", "movement_optimizer")


@pytest.mark.unit
@pytest.mark.parametrize("tile", TILES)
def test_real_checkout_tile_launch_target_exists(tile):
    if not inspect_tools_vendor_authority(REPO_ROOT).available:
        pytest.skip("vendor/ud-tools not materialised")
    model = ModelRegistry().get_model(tile)
    assert model is not None
    assert resolve_model_artifact(model, REPO_ROOT).is_file()
    assert resolve_tile_target(model, REPO_ROOT).resolvable
