"""Hermetic tests for scripts/requeue_stalled_merges.py (#2018). No network."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "requeue_stalled_merges", _SCRIPTS / "requeue_stalled_merges.py"
)
assert _SPEC and _SPEC.loader
requeue = importlib.util.module_from_spec(_SPEC)
sys.modules["requeue_stalled_merges"] = requeue
_SPEC.loader.exec_module(requeue)

pytestmark = pytest.mark.unit


def _done(stdout: str = "", rc: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        [], rc, stdout=stdout, stderr="boom" if rc else ""
    )


class FakeGh:
    """Replays the three REST reads the scanner makes."""

    def __init__(
        self,
        armed: Sequence[int],
        *,
        states: dict[int, str] | None = None,
        queue_events: dict[int, Sequence[str]] | None = None,
        list_rc: int = 0,
    ) -> None:
        self.armed = list(armed)
        self.states = states or {}
        self.queue_events = queue_events or {}
        self.list_rc = list_rc
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(cmd)
        self.calls.append(argv)
        joined = " ".join(argv)
        if "pulls?state=open" in joined:
            return _done("\n".join(map(str, self.armed)), self.list_rc)
        if "/timeline" in joined:
            pr = int(joined.split("/issues/")[1].split("/")[0])
            return _done("\n".join(self.queue_events.get(pr, ())))
        if "/pulls/" in joined:
            pr = int(joined.split("/pulls/")[1].split()[0])
            return _done(self.states.get(pr, "clean"))
        raise AssertionError(f"unexpected gh call: {joined}")


class Enqueuer:
    def __init__(self, fail: Sequence[int] = ()) -> None:
        self.fail = set(fail)
        self.seen: list[int] = []

    def __call__(self, repo: str, pr: int) -> tuple[bool, str]:
        assert repo == "o/r"
        self.seen.append(pr)
        return (pr not in self.fail, "boom" if pr in self.fail else "enqueued")


def test_only_clean_armed_unqueued_prs_are_enqueued() -> None:
    gh = FakeGh(
        [1, 2, 3, 4],
        states={2: "blocked", 3: "clean", 4: "unknown"},
        queue_events={3: ["added_to_merge_queue"]},
    )
    enq = Enqueuer()
    report = requeue.requeue_stalled("o/r", run=gh, enqueue=enq)
    assert enq.seen == [1]
    assert report.enqueued == (1,)
    assert report.failed == ()


def test_a_pr_removed_from_the_queue_counts_as_unqueued() -> None:
    gh = FakeGh(
        [5], queue_events={5: ["added_to_merge_queue", "removed_from_merge_queue"]}
    )
    enq = Enqueuer()
    requeue.requeue_stalled("o/r", run=gh, enqueue=enq)
    assert enq.seen == [5]


def test_the_open_list_is_one_rest_call_filtering_drafts_and_unarmed() -> None:
    gh = FakeGh([])
    requeue.requeue_stalled("o/r", run=gh, enqueue=Enqueuer())
    assert len(gh.calls) == 1
    call = " ".join(gh.calls[0])
    assert "graphql" not in call
    assert "select(.draft | not)" in call and "select(.auto_merge != null)" in call


def test_dry_run_reports_but_never_enqueues() -> None:
    gh = FakeGh([1, 2])
    enq = Enqueuer()
    report = requeue.requeue_stalled("o/r", run=gh, enqueue=enq, dry_run=True)
    assert enq.seen == []
    assert report.stalled == (1, 2)
    assert report.enqueued == ()


def test_at_most_limit_prs_are_enqueued_per_run() -> None:
    gh = FakeGh(list(range(1, 20)))
    enq = Enqueuer()
    report = requeue.requeue_stalled("o/r", run=gh, enqueue=enq, limit=3)
    assert enq.seen == [1, 2, 3]
    assert report.skipped_over_limit == 16


def test_a_failed_enqueue_is_reported_and_the_rest_still_run() -> None:
    gh = FakeGh([1, 2, 3])
    enq = Enqueuer(fail=[2])
    report = requeue.requeue_stalled("o/r", run=gh, enqueue=enq)
    assert enq.seen == [1, 2, 3]
    assert report.enqueued == (1, 3)
    assert report.failed == (2,)


def test_limit_and_repo_are_validated() -> None:
    with pytest.raises(ValueError):
        requeue.requeue_stalled("o/r", run=FakeGh([]), enqueue=Enqueuer(), limit=0)
    with pytest.raises(ValueError):
        requeue.requeue_stalled("not-a-slug", run=FakeGh([]), enqueue=Enqueuer())


def test_exit_code_is_nonzero_when_listing_or_an_enqueue_fails() -> None:
    ok = requeue.RequeueReport(stalled=(1,), enqueued=(1,), failed=())
    bad = requeue.RequeueReport(stalled=(1,), enqueued=(), failed=(1,))
    assert requeue.exit_code(ok) == 0
    assert requeue.exit_code(bad) == 1
    with pytest.raises(RuntimeError):
        requeue.requeue_stalled("o/r", run=FakeGh([], list_rc=1), enqueue=Enqueuer())


@pytest.mark.parametrize(
    "detail",
    [
        "PR is clean (enqueue failed: Pull request is already queued to merge)",
        "enqueue failed: This pull request is already in the merge queue",
    ],
)
def test_losing_the_race_to_githubs_own_enqueue_counts_as_success(
    detail: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub may enqueue the PR between the scan and our call; that is success."""

    class Result:
        armed = False

    result = Result()
    result.detail = detail  # type: ignore[attr-defined]
    monkeypatch.setattr(
        requeue.automerge_guard, "enqueue_stalled_pr", lambda repo, pr: result
    )
    assert requeue._guard_enqueue("o/r", 7) == (True, f"already queued: {detail}")


def test_a_real_enqueue_failure_is_still_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        armed = False
        detail = "enqueue failed: Resource not accessible by integration"

    monkeypatch.setattr(
        requeue.automerge_guard, "enqueue_stalled_pr", lambda repo, pr: Result()
    )
    ok, _ = requeue._guard_enqueue("o/r", 7)
    assert ok is False
