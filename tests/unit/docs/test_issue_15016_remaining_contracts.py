"""Contract tests for remaining #15016 SPEC notation, manifest honesty, and ch10 ZVCF."""

from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC_PATH = _REPO_ROOT / "SPEC.md"
_MANIFEST_PATH = _REPO_ROOT / "src" / "config" / "launcher_manifest.json"
_MIGRATION_PATH = _REPO_ROOT / "src" / "config" / "capability_migration.json"
_HELP_PATH = _REPO_ROOT / "docs" / "help" / "tools_calculator_hub.md"
_GOLF_SIM_HELP_PATH = _REPO_ROOT / "docs" / "help" / "golf_simulation_suite.md"
_CH10_PATH = (
    _REPO_ROOT
    / "docs"
    / "user_guide"
    / "upstream_drift_user_manual"
    / "ch10_opensim_physics_engine.md"
)


def _tile_by_id(manifest: dict, tile_id: str) -> dict:
    for tile in manifest["tiles"]:
        if tile["id"] == tile_id:
            return tile
    raise AssertionError(f"Tile {tile_id!r} not found in launcher_manifest.json")


def test_spec_section_4_uses_uppercase_control_input_map_G_of_x() -> None:
    """SPEC §4 must use G(x)u per CONTROL_AFFINE_TERMINOLOGY_CONTRACT / NOTATION.md."""
    spec_text = _SPEC_PATH.read_text(encoding="utf-8")
    section_match = re.search(
        r"### Instantaneous ZTCF / ZVCF Component Contract\n(.*?)\n### ",
        spec_text,
        re.DOTALL,
    )
    assert section_match is not None
    section = section_match.group(1)

    assert "xdot = f(x) + G(x)u" in section
    assert "ZVCF is `G(x)u`" in section
    assert "xdot = f(x) + g(x)u" not in section
    assert "ZVCF is `g(x)u`" not in section
    assert "lowercase `g(q)`" in section


def test_launcher_manifest_c5_honest_tile_capabilities_and_labels() -> None:
    """golf_simulation_suite and tools_calculator_hub labels/capabilities match behavior."""
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    migration = json.loads(_MIGRATION_PATH.read_text(encoding="utf-8"))
    help_text = _HELP_PATH.read_text(encoding="utf-8")
    golf_help_text = _GOLF_SIM_HELP_PATH.read_text(encoding="utf-8")

    golf_sim = _tile_by_id(manifest, "golf_simulation_suite")
    assert golf_sim["name"] == "Golf Simulation Suite"
    assert golf_sim["status"] == "prototype"
    assert "full_simulation" not in golf_sim["capabilities"]
    assert golf_sim["capabilities"] == ["prototype_visualization", "demo_inspection"]
    assert "prototype_visualization" in golf_help_text
    assert "demo_inspection" in golf_help_text

    calc_hub = _tile_by_id(manifest, "tools_calculator_hub")
    expected_name = "Tools Calculator Suite (Data Processor Entry)"
    assert calc_hub["name"] == expected_name
    assert migration["entries"]["tools_calculator_hub"]["name"] == expected_name
    assert "web_route" not in calc_hub
    assert calc_hub["capabilities"] == ["process_calculators", "data_processor_alias"]
    assert "(none; native-window)" in help_text
    assert "process_calculators, data_processor_alias" in help_text


def test_ch10_workflow_example_uses_canonical_zvcf_control_acceleration() -> None:
    """ch10 Complete Workflow Example uses compute_control_acceleration_at_state, not deprecated compute_zvcf."""
    ch10_text = _CH10_PATH.read_text(encoding="utf-8")
    workflow_match = re.search(
        r"## 10\.10 Complete Workflow Example\n(.*)",
        ch10_text,
        re.DOTALL,
    )
    assert workflow_match is not None
    workflow = workflow_match.group(1)

    assert "engine.compute_zvcf(" not in workflow
    assert "engine.compute_control_acceleration_at_state(q, tau)" in workflow

    code_match = re.search(r"```python\n(.*?)```", workflow, re.DOTALL)
    assert code_match is not None
    code_block = code_match.group(1)
    tree = ast.parse(code_block)
    attrs = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "engine"
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = OpenSimPhysicsEngine()
    assert [attr for attr in attrs if not hasattr(engine, attr)] == []
    assert "np.zeros_like(v)" in code_block
