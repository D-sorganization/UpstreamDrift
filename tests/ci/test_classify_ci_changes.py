"""Unit tests for scripts/classify_ci_changes.py (#13571 / UD-CI-010a)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.classify_ci_changes import (
    ChangeCategory,
    classify_path,
    is_doc_only_path,
    load_lane_config,
    path_matches_lane,
    should_run_opensim_lane,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "scripts/config/ci_opensim_path_patterns.json"


@pytest.fixture()
def config():
    return load_lane_config(CONFIG_PATH)


def test_path_fixture_includes_shared_physics_core(config: object) -> None:
    patterns = config.path_globs  # type: ignore[attr-defined]
    assert "src/shared/python/physics/**" in patterns
    assert path_matches_lane(
        "src/shared/python/physics/rust_kernel.py",
        patterns,
    )


def test_doc_only_change_does_not_run_lane(config: object) -> None:
    decision = should_run_opensim_lane(
        event_name="pull_request",
        changed_files=("docs/foo.md", "README.md"),
        labels=(),
        config=config,  # type: ignore[arg-type]
    )
    assert decision.run is False
    assert decision.reason == "doc_only"


def test_opensim_source_path_runs_lane(config: object) -> None:
    decision = should_run_opensim_lane(
        event_name="pull_request",
        changed_files=("src/engines/physics_engines/opensim/python/foo.py",),
        labels=(),
        config=config,  # type: ignore[arg-type]
    )
    assert decision.run is True
    assert decision.reason == "path"


def test_workflow_file_classifies_as_workflow(config: object) -> None:
    category = classify_path(
        ".github/workflows/ci-requires-opensim.yml",
        config,  # type: ignore[arg-type]
    )
    assert category is ChangeCategory.WORKFLOW


def test_label_without_matching_paths_still_runs(config: object) -> None:
    decision = should_run_opensim_lane(
        event_name="pull_request",
        changed_files=("src/tools/foo.py",),
        labels=("ci:opensim",),
        config=config,  # type: ignore[arg-type]
    )
    assert decision.run is True
    assert decision.reason == "label"


def test_non_pr_events_always_run(config: object) -> None:
    for event in ("push", "schedule", "workflow_dispatch"):
        decision = should_run_opensim_lane(
            event_name=event,
            changed_files=(),
            labels=(),
            config=config,  # type: ignore[arg-type]
        )
        assert decision.run is True
        assert decision.reason == event


def test_is_doc_only_path_respects_license(config: object) -> None:
    doc_globs = config.doc_only_globs  # type: ignore[attr-defined]
    assert is_doc_only_path("LICENSE", doc_globs)
    assert not is_doc_only_path("src/foo.py", doc_globs)


def test_config_json_is_valid_fixture() -> None:
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert raw["label"] == "ci:opensim"
    assert raw["path_globs"]


def test_github_output_uses_os_environ(tmp_path, monkeypatch) -> None:
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    from scripts.classify_ci_changes import main

    assert (
        main(
            [
                "opensim",
                "--event",
                "pull_request",
                "--changed-file",
                "docs/foo.md",
                "--github-output",
            ]
        )
        == 0
    )
    text = output_file.read_text(encoding="utf-8")
    assert "run=false" in text
    assert "reason=doc_only" in text
