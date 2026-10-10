"""Heavy optional CI lanes run nightly or on demand, not on every PR push (#12224).

The docker fleet lane is the scarcest runner capacity. Cross-engine
equivalence, the optional stack and the PR leaderboard artifact used to run on
every PR push; with per-ref ``cancel-in-progress`` almost none of them ever
completed. These tests pin the gating contract:

* the light conformance gate still runs on every PR and main push;
* heavy jobs run on ``schedule``, ``workflow_dispatch`` or a PR carrying the
  ``ci:full`` label;
* label events other than ``ci:full`` neither start work nor cancel a run;
* non-PR runs are never cancelled, and a nightly failure files an issue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
FULL_LABEL = "ci:full"


def _load(name: str) -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{name} did not parse to a mapping"
    return data


def _triggers(data: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 1.1 parses the bare key ``on`` as boolean True.
    return data.get("on", data.get(True, {}))


def _only_full_label_events_start(condition: str) -> None:
    assert "github.event.action != 'labeled'" in condition
    assert f"github.event.label.name == '{FULL_LABEL}'" in condition


@pytest.mark.parametrize(
    "name",
    [
        "cross-engine-equivalence.yml",
        "ci-optional-stack.yml",
        "cross-engine-leaderboard-publish.yml",
    ],
)
def test_other_label_events_cannot_cancel_a_run(name: str) -> None:
    group = _load(name)["concurrency"]["group"]
    assert "github.ref" in group
    assert f"github.event.label.name != '{FULL_LABEL}'" in group


class TestCrossEngineEquivalenceGating:
    NAME = "cross-engine-equivalence.yml"

    def test_triggers_include_label_and_nightly_schedule(self) -> None:
        on = _triggers(_load(self.NAME))
        assert "labeled" in on["pull_request"]["types"]
        assert on["schedule"], "heavy lanes need a nightly schedule"
        assert "workflow_dispatch" in on

    def test_only_pr_runs_are_cancelled_and_labels_do_not_cancel(self) -> None:
        concurrency = _load(self.NAME)["concurrency"]
        assert concurrency["cancel-in-progress"] == (
            "${{ github.event_name == 'pull_request' }}"
        )
        assert "github.sha" in concurrency["group"]

    def test_dispatcher_exposes_heavy_flag(self) -> None:
        dispatcher = _load(self.NAME)["jobs"]["pick-runner"]
        _only_full_label_events_start(dispatcher["if"])
        heavy = dispatcher["outputs"]["heavy"]
        for clause in (
            "github.event_name == 'schedule'",
            "github.event_name == 'workflow_dispatch'",
            f"contains(github.event.pull_request.labels.*.name, '{FULL_LABEL}')",
        ):
            assert clause in heavy

    @pytest.mark.parametrize("job", ["equivalence", "force-overlay-parity"])
    def test_heavy_jobs_require_heavy_flag(self, job: str) -> None:
        condition = _load(self.NAME)["jobs"][job]["if"]
        assert "needs.pick-runner.outputs.heavy == 'true'" in condition

    def test_conformance_gate_stays_on_every_pr(self) -> None:
        job = _load(self.NAME)["jobs"]["canonical-conformance-gate"]
        assert "heavy" not in job["if"]

    def test_nightly_failure_files_issue(self) -> None:
        job = _load(self.NAME)["jobs"]["report-nightly"]
        assert set(job["needs"]) == {"equivalence", "force-overlay-parity"}
        assert "always()" in job["if"]
        assert "github.event_name == 'schedule'" in job["if"]
        assert job["permissions"] == {"issues": "write"}
        script = job["steps"][-1]["with"]["script"]
        assert "issues.create" in script
        assert "state: 'closed'" in script


class TestOptionalStackGating:
    NAME = "ci-optional-stack.yml"

    def test_runs_nightly_not_on_every_push(self) -> None:
        on = _triggers(_load(self.NAME))
        assert "push" not in on
        assert "labeled" in on["pull_request"]["types"]
        assert on["schedule"][0]["cron"].split()[-1] == "*", "must run daily"

    def test_pr_runs_need_full_label(self) -> None:
        condition = _load(self.NAME)["jobs"]["pick-runner"]["if"]
        _only_full_label_events_start(condition)
        assert f"contains(github.event.pull_request.labels.*.name, '{FULL_LABEL}')" in (
            condition
        )


class TestLeaderboardPublishGating:
    NAME = "cross-engine-leaderboard-publish.yml"

    def test_pr_artifact_needs_full_label_and_main_publish_unchanged(self) -> None:
        data = _load(self.NAME)
        on = _triggers(data)
        assert on["push"]["branches"] == ["main"]
        assert "labeled" in on["pull_request"]["types"]
        condition = data["jobs"]["publish"]["if"]
        assert "github.event_name != 'pull_request'" in condition
        assert f"contains(github.event.pull_request.labels.*.name, '{FULL_LABEL}')" in (
            condition
        )
