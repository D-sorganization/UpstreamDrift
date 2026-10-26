"""Contracts for fast-tier pytest module selection (#13578)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.pytest_fast_tier_module_selection import (
    SelectionConfig,
    load_config,
    select_modules,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_full_tier_always_collects_entire_green_suite() -> None:
    decision = select_modules(
        tier="full",
        changed_files=("tests/ci/test_pytest_fast_tier_module_selection.py",),
    )
    assert decision.mode == "full"
    assert decision.modules == ()


def test_shared_core_change_falls_back_to_full_collection() -> None:
    config = load_config()
    decision = select_modules(
        tier="fast",
        changed_files=("src/shared/python/physics/rust_kernel.py",),
        config=config,
    )
    assert decision.mode == "full"
    assert decision.reason == "shared_core_or_workflow"


def test_changed_ci_script_selects_its_contract_test_module() -> None:
    decision = select_modules(
        tier="fast",
        changed_files=("scripts/classify_ci_changes.py",),
        repo_root=REPO_ROOT,
    )
    assert decision.mode == "modules"
    assert "tests/ci/test_classify_ci_changes.py" in decision.modules


def test_scripts_ci_change_falls_back_to_full_green_suite() -> None:
    decision = select_modules(
        tier="fast",
        changed_files=("scripts/ci/classify_event_tier.py",),
        repo_root=REPO_ROOT,
    )
    assert decision.mode == "full"
    assert decision.reason == "shared_core_or_workflow"


def test_changed_test_file_selects_only_that_module(tmp_path: Path) -> None:
    test_file = tmp_path / "tests" / "unit" / "demo" / "test_sample.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_ok() -> None:\n    assert True\n", encoding="utf-8")
    decision = select_modules(
        tier="fast",
        changed_files=("tests/unit/demo/test_sample.py",),
        repo_root=tmp_path,
        config=SelectionConfig(shared_core_globs=("src/shared/**",), workflow_globs=()),
    )
    assert decision.mode == "modules"
    assert decision.modules == ("tests/unit/demo/test_sample.py",)


def test_workflow_wires_fast_tier_selection_on_pull_requests() -> None:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/ci-standard.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["unit-test-gate"]
    run_step = next(
        step for step in job["steps"] if step.get("name") == "Run Green-Suite Unit Gate"
    )
    run = run_step["run"]
    assert "pytest_fast_tier_module_selection.py" in run
    assert run_step["env"]["CI_TIER"] == "${{ needs.changed-paths.outputs.tier }}"
    assert 'github.event_name }}" = "pull_request"' in run


def test_tracked_src_change_also_selects_monolith_register_test(
    tmp_path: Path,
) -> None:
    """A src/ edit can push a file over the LOC threshold (#13842)."""
    for rel in (
        "tests/unit/test_widget.py",
        "tests/scripts/test_monolith_register.py",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("", encoding="utf-8")
    decision = select_modules(
        tier="fast",
        changed_files=("src/engines/widget.py",),
        repo_root=tmp_path,
        config=SelectionConfig(("pyproject.toml",), ()),
    )
    assert decision.mode == "modules"
    assert "tests/scripts/test_monolith_register.py" in decision.modules
    assert "tests/unit/test_widget.py" in decision.modules


def test_register_doc_change_selects_monolith_register_test(tmp_path: Path) -> None:
    for rel in ("tests/unit/test_a.py", "tests/scripts/test_monolith_register.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("", encoding="utf-8")
    decision = select_modules(
        tier="fast",
        changed_files=(
            "tests/unit/test_a.py",
            "docs/development/monolith_refactor_register.md",
        ),
        repo_root=tmp_path,
        config=SelectionConfig(("pyproject.toml",), ()),
    )
    assert decision.mode == "modules"
    assert "tests/scripts/test_monolith_register.py" in decision.modules


def test_unrelated_change_does_not_add_monolith_register_test(tmp_path: Path) -> None:
    for rel in ("tests/unit/test_a.py", "tests/scripts/test_monolith_register.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("", encoding="utf-8")
    decision = select_modules(
        tier="fast",
        changed_files=("tests/unit/test_a.py",),
        repo_root=tmp_path,
        config=SelectionConfig(("pyproject.toml",), ()),
    )
    assert decision.modules == ("tests/unit/test_a.py",)
