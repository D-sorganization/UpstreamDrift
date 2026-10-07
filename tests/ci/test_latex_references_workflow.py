"""Contract for the workflow that typesets docs/research LaTeX references (#11574)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import fork_pr_runner_guard as guard

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/latex-references.yml"


def _data() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _job() -> dict[str, Any]:
    return _data()["jobs"]["compile"]


def _step(name: str) -> dict[str, Any]:
    return next(s for s in _job()["steps"] if s.get("name") == name)


def test_triggers_on_research_tex_changes_only() -> None:
    triggers = _data()[True]  # PyYAML parses the bare key ``on`` as True
    assert "docs/research/**/*.tex" in triggers["pull_request"]["paths"]


def test_job_has_timeout_concurrency_and_fork_guard() -> None:
    data = _data()
    assert data["concurrency"]["cancel-in-progress"] is True
    assert isinstance(_job()["timeout-minutes"], int)
    assert "head.repo.full_name == github.repository" in _job()["if"]


def test_fork_pr_runner_guard_accepts_workflow(tmp_path: Path) -> None:
    (tmp_path / WORKFLOW.name).write_text(
        WORKFLOW.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert guard.find_violations(tmp_path) == []


def test_actions_are_pinned_to_full_shas() -> None:
    for step in _job()["steps"]:
        if "uses" in step:
            assert re.search(r"@[0-9a-f]{40}$", step["uses"]), step["uses"]


def test_tectonic_is_pinned_with_verified_checksum() -> None:
    env = _data()["env"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", env["TECTONIC_VERSION"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["TECTONIC_SHA256"])
    assert "sha256sum -c" in _step("Install pinned Tectonic")["run"]


def test_run_blocks_do_not_interpolate_untrusted_expressions() -> None:
    for step in _job()["steps"]:
        assert "${{" not in str(step.get("run", "")), step.get("name")


def test_title_case_check_runs_on_changed_files_only() -> None:
    run = _step("Document title case (changed files only)")["run"]
    assert "check_document_title_case.py" in run
    assert "changed_tex.txt" in run


def test_pdfs_are_uploaded_and_missing_pdfs_fail() -> None:
    step = _step("Upload PDFs")
    assert step["with"]["if-no-files-found"] == "error"


def test_every_buildable_reference_has_a_matching_main_file() -> None:
    mains = sorted((REPO_ROOT / "docs/research").glob("*/*.tex"))
    assert mains
    for main in mains:
        assert main.stem == main.parent.name, main
