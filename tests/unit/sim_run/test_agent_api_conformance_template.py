"""Conformance template stays aligned with pinned ``ud-sim-mcp`` tools (#15022)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sim_run.mcp import SimulationMcpServer

pytestmark = pytest.mark.unit

_TEMPLATE = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "config"
    / "fleet_agent_api_conformance"
    / "v1"
    / "template.json"
)


def test_conformance_template_required_tools_match_pinned_specs() -> None:
    payload = json.loads(_TEMPLATE.read_text(encoding="utf-8"))
    required = payload.get("required_tools")
    if not isinstance(required, list):
        raise TypeError("required_tools must be a list")
    backend = SimulationMcpServer()
    pinned = {spec["name"] for spec in backend.tool_specs()}
    assert set(required) == pinned


def test_conformance_deliverables_present() -> None:
    """Issue acceptance artifacts from #15022 remain on disk."""
    repo_root = Path(__file__).resolve().parents[3]
    paths = (
        repo_root / ".github" / "workflows" / "agent-api-conformance.yml",
        _TEMPLATE,
        repo_root / "src" / "sim_run" / "mcp_fastmcp.py",
        repo_root / "llms.txt",
        repo_root / "scripts" / "run_agent_api_conformance.py",
    )
    for path in paths:
        assert path.is_file(), f"missing {path.relative_to(repo_root)}"
