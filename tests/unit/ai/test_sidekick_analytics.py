"""Tests for the Sidekick analytics tool (issue #5464)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.tools import sidekick_analytics


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configure the tool to read runs from ``tmp_path`` and return it."""
    monkeypatch.setenv("UPSTREAMDRIFT_SIM_RUNS_DIR", str(tmp_path))
    return tmp_path


def _write_manifest(
    runs_dir: Path,
    run_id: str,
    body: dict,
) -> Path:
    """Create a manifest at ``<runs_dir>/<run_id>/manifest.json``."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


# ── Happy path ──────────────────────────────────────────────────────


def test_summary_structure_for_valid_run(runs_dir: Path) -> None:
    """The tool returns the documented dict shape."""
    _write_manifest(
        runs_dir,
        "run_001",
        {
            "engine": "mujoco",
            "duration_s": 1.234,
            "n_frames": 300,
            "key_metrics": {"peak_torque": 42.0, "club_speed": 38.7},
        },
    )

    result = sidekick_analytics.summarize_simulation_run("run_001")

    assert result["run_id"] == "run_001"
    assert result["engine"] == "mujoco"
    assert result["duration_s"] == pytest.approx(1.234)
    assert result["n_frames"] == 300
    assert result["key_metrics"] == {"peak_torque": 42.0, "club_speed": 38.7}
    assert isinstance(result["summary"], str)
    assert "run_001" in result["summary"]
    assert "mujoco" in result["summary"]
    # Key metrics surface in the deterministic summary.
    assert "peak_torque" in result["summary"]


def test_summary_handles_missing_optional_fields(runs_dir: Path) -> None:
    """Missing duration/frames/metrics gracefully degrade to None / {}."""
    _write_manifest(runs_dir, "run_minimal", {"engine": "drake"})

    result = sidekick_analytics.summarize_simulation_run("run_minimal")

    assert result["engine"] == "drake"
    assert result["duration_s"] is None
    assert result["n_frames"] is None
    assert result["key_metrics"] == {}
    assert "no key metrics recorded" in result["summary"]


# ── Validation failures ─────────────────────────────────────────────


def test_missing_run_raises_value_error(runs_dir: Path) -> None:
    """Unknown run ids raise a clear ``ValueError``."""
    with pytest.raises(ValueError, match="Unknown run_id"):
        sidekick_analytics.summarize_simulation_run("does_not_exist")


def test_path_injection_rejected(runs_dir: Path) -> None:
    """``run_id`` containing path separators is rejected."""
    with pytest.raises(ValueError, match="path separators"):
        sidekick_analytics.summarize_simulation_run("../etc/passwd")


def test_backslash_path_rejected(runs_dir: Path) -> None:
    """Windows-style separators are rejected too."""
    with pytest.raises(ValueError, match="path separators"):
        sidekick_analytics.summarize_simulation_run("..\\evil")


def test_empty_run_id_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        sidekick_analytics.summarize_simulation_run("")
    with pytest.raises(ValueError, match="non-empty"):
        sidekick_analytics.summarize_simulation_run("   ")


def test_non_string_run_id_rejected() -> None:
    with pytest.raises(TypeError, match="run_id must be a string"):
        sidekick_analytics.summarize_simulation_run(42)  # type: ignore[arg-type]


def test_parent_dir_token_rejected() -> None:
    with pytest.raises(ValueError):
        sidekick_analytics.summarize_simulation_run("..")


def test_malformed_manifest_raises(runs_dir: Path) -> None:
    """Invalid JSON in the manifest surfaces as a ValueError."""
    run_dir = runs_dir / "broken"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        sidekick_analytics.summarize_simulation_run("broken")


# ── Registry hookup ─────────────────────────────────────────────────


def test_tool_registers_with_registry() -> None:
    """``register_sidekick_analytics_tools`` adds the expected tool."""
    registry = ToolRegistry()
    sidekick_analytics.register_sidekick_analytics_tools(registry)
    assert "summarize_simulation_run" in registry
    tool = registry.get_tool("summarize_simulation_run")
    assert tool is not None
    assert tool.handler is sidekick_analytics.summarize_simulation_run


def test_register_rejects_non_registry() -> None:
    with pytest.raises(TypeError, match="ToolRegistry"):
        sidekick_analytics.register_sidekick_analytics_tools(object())  # type: ignore[arg-type]


def test_global_golf_suite_registration_includes_sidekick() -> None:
    """``register_golf_suite_tools`` wires the Sidekick analytics tool."""
    from src.shared.python.ai.sample_tools import register_golf_suite_tools

    registry = ToolRegistry()
    register_golf_suite_tools(registry)
    assert "summarize_simulation_run" in registry


# ── System prompt mention ───────────────────────────────────────────


def test_system_prompt_mentions_summarize_tool() -> None:
    """``build_system_prompt`` includes a hint pointing at the new tool."""
    from src.shared.python.ai.system_prompts import build_system_prompt

    prompt = build_system_prompt(app_context="upstream_drift")
    assert "summarize_simulation_run" in prompt
