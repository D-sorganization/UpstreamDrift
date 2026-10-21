"""Heavy optional CI lanes run nightly or on demand, not on every PR push (#12224).

The docker fleet lane is the scarcest runner capacity. Cross-engine
equivalence, the optional stack and the PR leaderboard artifact used to run on
every PR push; with per-ref ``cancel-in-progress`` almost none of them ever
completed. These tests pin the gating contract:

* the light conformance gate still runs on every PR and main push;
* heavy jobs run on ``schedule``, ``workflow_dispatch`` or a PR carrying the
  ``ci:full`` label;
* label events other than ``ci:full`` neither start work nor cancel a run;
* non-PR runs are never cancelled once running, only the newest pending one
  is kept (#12285), and a nightly failure files an issue.
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

    def test_only_pr_runs_cancel_in_progress(self) -> None:
        concurrency = _load(self.NAME)["concurrency"]
        assert (
            concurrency["cancel-in-progress"]
            == "${{ github.event_name == 'pull_request' }}"
        )

    def test_non_pr_runs_keep_only_the_newest_pending_run(self) -> None:
        # #12285: a per-SHA group queued one main run per push and never
        # dropped the stale ones. Grouped per event and ref without
        # cancel-in-progress, GitHub keeps the running run plus only the
        # newest pending one, and a push never displaces the nightly run.
        group = _load(self.NAME)["concurrency"]["group"]
        assert "github.sha" not in group
        assert "format('{0}-{1}', github.event_name, github.ref)" in group

    def test_workflow_lint_allows_conditional_cancel(self) -> None:
        # The workflow-file lint greps for a literal ``cancel-in-progress:
        # true``, so the conditional form above must be on its exception list.
        lint = (WORKFLOWS / "lint-workflow-files.yml").read_text(encoding="utf-8")
        allowlist = next(
            line for line in lint.splitlines() if "nightly-publish.yml" in line
        )
        assert self.NAME in allowlist

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


class TestOpenSimMainRunsComplete:
    """#13494: a per-SHA group queued one OpenSim run for every main push.

    ``ci-requires-opensim.yml`` keyed its main group on ``github.sha``, so no
    main run ever superseded another and 42 main pushes sat queued for the four
    docker runners. Grouped per event and ref without cancel-in-progress,
    GitHub keeps the running main run plus only the newest pending one. PR runs
    still cancel superseded pushes.

    #13513: the job is not a required check (only ``quality-gate`` is), so the
    merge queue never waited on it. It no longer triggers on ``merge_group``;
    PR runs give pre-merge signal and main push runs give post-merge signal.

    #13571: path/label gating moved into ``classify-changes``; nightly schedule
    and ``ci:opensim`` label events follow the same concurrency pattern as the
    optional-stack lane.
    """

    NAME = "ci-requires-opensim.yml"
    CANCEL = "${{ github.event_name == 'pull_request' }}"
    OPENSIM_LABEL = "ci:opensim"

    def test_group_is_per_event_and_ref(self) -> None:
        group = _load(self.NAME)["concurrency"]["group"]
        assert "github.sha" not in group, "per-SHA groups queue every main push"
        assert "runner." not in group, "runner context is invalid in concurrency"
        assert f"github.event.label.name != '{self.OPENSIM_LABEL}'" in group

    def test_only_pr_runs_cancel_in_progress(self) -> None:
        # push and dispatch runs evaluate this to false, so a main run that
        # has started is never cancelled.
        concurrency = _load(self.NAME)["concurrency"]
        assert concurrency["cancel-in-progress"] == self.CANCEL

    def test_no_merge_group_trigger(self) -> None:
        # Not a required check, so merge-queue runs were pure runner cost.
        on = _triggers(_load(self.NAME))
        assert "merge_group" not in on, "merge queue never waits on this job"

    def test_pr_paths_or_label_gating_is_job_level(self) -> None:
        on = _triggers(_load(self.NAME))
        pr = on["pull_request"]
        assert "paths" not in pr
        assert "paths-ignore" not in pr
        assert "labeled" in pr["types"]
        jobs = _load(self.NAME)["jobs"]
        assert "classify-changes" in jobs
        assert "needs.classify-changes.outputs.run == 'true'" in jobs["pick-runner"]["if"]

    def test_post_merge_nightly_and_dispatch_triggers_remain(self) -> None:
        on = _triggers(_load(self.NAME))
        assert on["push"]["branches"] == ["main"], "main runs give post-merge signal"
        assert on["schedule"], "nightly covers non-OpenSim PR regressions"
        assert "workflow_dispatch" in on

    def test_workflow_lint_allows_conditional_cancel(self) -> None:
        lint = (WORKFLOWS / "lint-workflow-files.yml").read_text(encoding="utf-8")
        allowlist = next(
            line for line in lint.splitlines() if "nightly-publish.yml" in line
        )
        assert self.NAME in allowlist
