"""Workflow contract for ci-requires-opensim.yml (#13571 / UD-CI-002)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.classify_ci_changes import load_lane_config, should_run_opensim_lane

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/ci-requires-opensim.yml"
CONFIG_PATH = REPO_ROOT / "scripts/config/ci_opensim_path_patterns.json"
OPENSIM_LABEL = "ci:opensim"


def _load() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _triggers(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("on", data.get(True, {}))


class TestOpenSimWorkflowTriggers:
    def test_pull_request_has_no_path_filter(self) -> None:
        """Path gating lives in classify-changes, not on pull_request (#13571)."""
        pr = _triggers(_load())["pull_request"]
        assert "paths" not in pr
        assert "paths-ignore" not in pr
        assert "labeled" in pr["types"]

    def test_nightly_schedule_and_dispatch_present(self) -> None:
        on = _triggers(_load())
        assert on["schedule"]
        assert "workflow_dispatch" in on

    def test_no_merge_group_trigger(self) -> None:
        assert "merge_group" not in _triggers(_load())

    def test_main_push_post_merge_signal_retained(self) -> None:
        on = _triggers(_load())
        assert on["push"]["branches"] == ["main"]


class TestOpenSimJobLevelGating:
    def test_classifier_job_feeds_heavy_lane(self) -> None:
        jobs = _load()["jobs"]
        assert "classify-changes" in jobs
        classify = jobs["classify-changes"]
        assert "outputs" in classify
        assert "run" in classify["outputs"]

        pick = jobs["pick-runner"]["if"]
        assert "needs.classify-changes.outputs.run == 'true'" in pick

        opensim = jobs["requires-opensim"]["if"]
        assert "needs.classify-changes.outputs.run != 'true'" in opensim
        assert "needs.pick-runner.result == 'success'" in opensim

    def test_doc_only_pr_classifier_skips_lane(self) -> None:
        config = load_lane_config(CONFIG_PATH)
        decision = should_run_opensim_lane(
            event_name="pull_request",
            changed_files=("docs/plans/foo.md",),
            labels=(),
            config=config,
        )
        assert decision.run is False

    def test_labeled_pr_outside_paths_still_runs(self) -> None:
        config = load_lane_config(CONFIG_PATH)
        decision = should_run_opensim_lane(
            event_name="pull_request",
            changed_files=("src/api/routes/health.py",),
            labels=(OPENSIM_LABEL,),
            config=config,
        )
        assert decision.run is True
        assert decision.reason == "label"

    def test_synchronize_while_labeled_matches_label_reason(self) -> None:
        config = load_lane_config(CONFIG_PATH)
        decision = should_run_opensim_lane(
            event_name="pull_request",
            changed_files=("docs/only.md",),
            labels=(OPENSIM_LABEL,),
            config=config,
        )
        assert decision.run is True

    def test_heavy_steps_guarded_when_lane_skipped(self) -> None:
        steps = _load()["jobs"]["requires-opensim"]["steps"]
        models = next(
            s for s in steps if s.get("name") == "Fetch OpenSim Rajagopal body model"
        )
        assert "needs.classify-changes.outputs.run == 'true'" in str(
            models.get("if", "")
        )

    def test_skip_step_reports_success_for_required_check(self) -> None:
        steps = _load()["jobs"]["requires-opensim"]["steps"]
        skip = next(s for s in steps if "Skip OpenSim lane" in str(s.get("name", "")))
        assert skip
        assert "needs.classify-changes.outputs.run != 'true'" in str(skip.get("if", ""))

    def test_workflow_path_list_matches_config_fixture(self) -> None:
        config = load_lane_config(CONFIG_PATH)
        workflow_text = WORKFLOW.read_text(encoding="utf-8")
        assert "src/shared/python/physics/**" in json.dumps(config.path_globs)
        assert "classify_ci_changes.py" in workflow_text

    def test_regression_policy_documented_in_workflow(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "nightly" in text.lower()
        assert "@core" in text or "core" in text.lower()
        assert "24" in text

    def test_lightweight_jobs_reuse_reversible_dispatch_policy(self) -> None:
        jobs = _load()["jobs"]
        route = jobs["pick-runner"]["runs-on"]
        assert "!github.event.repository.private" in route
        assert "vars.CI_RUNNER_MODE != 'local'" in route
        assert jobs["classify-changes"]["runs-on"] == route
        assert jobs["report-nightly"]["runs-on"] == "d-sorg-fleet-docker"
        real_route = jobs["requires-opensim"]["runs-on"]
        assert "needs.classify-changes.outputs.run == 'true'" in real_route
        assert "needs.pick-runner.outputs.runner" in real_route
        assert route.removeprefix("${{").removesuffix("}}").strip() in real_route
        assert "runner=d-sorg-fleet-docker" in jobs["pick-runner"]["steps"][0]["run"]
        for name in ("classify-changes", "pick-runner", "requires-opensim"):
            assert "head.repo.full_name == github.repository" in jobs[name]["if"]

    def test_failed_diff_never_reports_skip(self, tmp_path: Path) -> None:
        steps = _load()["jobs"]["classify-changes"]["steps"]
        script = next(step["run"] for step in steps if step.get("id") == "classify")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        for name, body in {
            "git": "exit 73",
            "python3": 'touch "$CLASSIFIER_CALLED"',
        }.items():
            command = bindir / name
            command.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
            command.chmod(0o755)
        called = tmp_path / "called"
        output = tmp_path / "output"
        result = subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "PATH": f"{bindir}:{os.environ['PATH']}",
                "GITHUB_OUTPUT": str(output),
                "EVENT_NAME": "pull_request",
                "BASE_REF": "missing",
                "PR_LABELS": "",
                "CLASSIFIER_CALLED": str(called),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 73
        assert not called.exists()
        assert not output.exists() or "run=false" not in output.read_text()

    def test_pytest_step_adjudicates_ci_lane_quarantine(self) -> None:
        steps = _load()["jobs"]["requires-opensim"]["steps"]
        pytest_step = next(
            s for s in steps if s.get("name") == "Run requires_opensim tests"
        )
        body = str(pytest_step.get("run", ""))
        assert "scripts.ci.ci_lane_quarantine" in body
        assert "ci-requires-opensim.yml" in body
        assert "Smoke squat joint-reaction acceleration gate" in {
            s.get("name") for s in steps
        }
