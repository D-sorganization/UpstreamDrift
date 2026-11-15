"""Tests for user-guide, examples, knowledge-pack, and README alignment (#15016)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_installation_guide_uses_current_org_submodule_and_pyproject() -> None:
    text = (REPO_ROOT / "docs/user_guide/installation.md").read_text(encoding="utf-8")
    for stale in (
        "dieterolson/UpstreamDrift",
        "Golf_Modeling_Suite",
        "git lfs",
        "requirements.txt",
        "MATLAB R2023a",
    ):
        assert stale not in text, f"Stale reference {stale!r} still in installation.md"

    for required in (
        "D-sorganization/UpstreamDrift",
        "vendor/ud-tools",
        'pip install -e ".[dev]"',
        "MATLAB R2025b",
    ):
        assert required in text, f"Expected {required!r} in installation.md"


def test_getting_started_mentions_ud_tools_submodule() -> None:
    text = (REPO_ROOT / "docs/user_guide/getting_started.md").read_text(
        encoding="utf-8"
    )
    assert "vendor/ud-tools" in text


def test_user_guide_entry_points_and_branding_updated() -> None:
    for rel in (
        "docs/user_guide/README.md",
        "docs/user_guide/running_simulations.md",
        "docs/user_guide/launchers.md",
        "examples/README.md",
    ):
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "Golf Modeling Suite" not in text, f"Stale branding in {rel}"
        assert "Golf_Modeling_Suite" not in text, f"Stale repo name in {rel}"

    running_sim = (REPO_ROOT / "docs/user_guide/running_simulations.md").read_text(
        encoding="utf-8"
    )
    assert "python launch_golf_suite.py" not in running_sim
    assert "from shared.python.output_manager" not in running_sim
    assert "launch_upstream_drift.py" in running_sim or "upstream-drift" in running_sim

    launchers = (REPO_ROOT / "docs/user_guide/launchers.md").read_text(encoding="utf-8")
    assert "python launch_golf_suite.py" not in launchers
    for expected in (
        "launch_upstream_drift.py",
        "upstream-drift",
        "sidekick",
        "ud-sim",
        "--install-shortcuts",
        "--engine",
        "--port",
        "--no-browser",
    ):
        assert expected in launchers, f"Expected {expected!r} in launchers.md"


def test_examples_readme_covers_top_level_example_scripts() -> None:
    readme = (REPO_ROOT / "examples/README.md").read_text(encoding="utf-8")
    top_level_scripts = sorted(
        p.name for p in (REPO_ROOT / "examples").glob("*.py") if p.name != "__init__.py"
    )
    for script_name in top_level_scripts:
        assert script_name in readme, (
            f"Missing example {script_name} in examples/README.md"
        )
    assert "motion_capture/" in readme
    assert "simulation_backends/" in readme


def test_knowledge_pack_includes_conventions_engines_physics_and_spec() -> None:
    pack = yaml.safe_load(
        (REPO_ROOT / "knowledge/pack.yml").read_text(encoding="utf-8")
    )
    includes: list[str] = []
    for source in pack.get("sources", []):
        includes.extend(source.get("include", []))

    for pattern in (
        "SPEC.md",
        "docs/conventions/**/*.md",
        "docs/engines/**/*.md",
        "docs/simulation_backends/**/*.md",
        "docs/physics/**/*.md",
    ):
        assert pattern in includes, f"Expected {pattern!r} in knowledge/pack.yml"


def test_root_readme_documents_cli_scripts_flags_and_additional_engines() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for expected in (
        "sidekick",
        "ud-sim",
        "ud-sim-mcp",
        "feature_registry",
        "--port",
        "--no-browser",
        "--install-shortcuts",
        "jaxsim",
        "pendulum",
        "putting_green",
    ):
        assert expected in readme, f"Expected {expected!r} in README.md"
