"""Hermetic tests for scripts/automerge_guard.py — no network, no real gh."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

# scripts/ is not an importable package, so load the module by path. It must be
# registered in sys.modules before exec_module or @dataclass cannot resolve its
# own module namespace.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "automerge_guard.py"
_SPEC = importlib.util.spec_from_file_location("automerge_guard", _MODULE_PATH)
assert _SPEC and _SPEC.loader
automerge_guard = importlib.util.module_from_spec(_SPEC)
sys.modules["automerge_guard"] = automerge_guard
_SPEC.loader.exec_module(automerge_guard)

pytestmark = pytest.mark.unit

# Fake token strings only (#1917). Never use a real credential in a test.
FAKE_APP_TOKEN = "ghs_" + "A" * 36
FAKE_CLASSIC_PAT = "ghp_" + "B" * 36
FAKE_FINE_GRAINED_PAT = "github_pat_" + "C" * 82
FAKE_OAUTH_TOKEN = "gho_" + "D" * 36
FAKE_USER_TO_SERVER = "ghu_" + "E" * 36


@pytest.fixture(autouse=True)
def _app_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep main() hermetic: the identity check reads GH_TOKEN, not real gh."""
    monkeypatch.setenv("GH_TOKEN", FAKE_APP_TOKEN)


HEAD_SHA = "abc123"
HEAD_DATE = "2026-08-13T20:35:02Z"


def _completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class FakeGh:
    """Records gh invocations and replays canned responses."""

    def __init__(
        self,
        *,
        draft: bool = False,
        labels: Sequence[str] = (),
        body: str = "",
        disarms: Sequence[str] = (),
        removed: Sequence[str] = (),
        head_date: str = HEAD_DATE,
        committer_date: str = HEAD_DATE,
        force_pushes: Sequence[str] = (),
        force_push_fails: bool = False,
        merge_returncode: int = 0,
    ) -> None:
        self.pull = {
            "draft": draft,
            "labels": [{"name": name} for name in labels],
            "body": body,
            "head": {
                "sha": HEAD_SHA,
                "ref": "feat/x",
                "repo": {"full_name": "o/r"},
            },
            "base": {"ref": "main", "repo": {"full_name": "o/r"}},
            "node_id": "PR_node7",
        }
        self.disarms = list(disarms)
        self.removed = list(removed)
        self.head_date = head_date
        self.committer_date = committer_date
        self.force_pushes = list(force_pushes)
        self.force_push_fails = force_push_fails
        self.merge_returncode = merge_returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(cmd)
        self.calls.append(argv)
        joined = " ".join(argv)
        if "pr" in argv and "merge" in argv:
            return _completed(
                returncode=self.merge_returncode,
                stderr="merge boom" if self.merge_returncode else "",
            )
        if "/timeline" in joined and "head_ref_force_pushed" in joined:
            if self.force_push_fails:
                return _completed(returncode=1, stderr="timeline boom")
            return _completed("\n".join(self.force_pushes))
        if "/timeline" in joined:
            return _completed("\n".join(self.disarms))
        if "/check-suites" in joined:
            return _completed(self.head_date)
        if "/commits/" in joined:
            return _completed(self.committer_date)
        if "/files" in joined:
            return _completed("\n".join(self.removed))
        if "/pulls/" in joined:
            return _completed(json.dumps(self.pull))
        raise AssertionError(f"unexpected gh call: {joined}")

    @property
    def armed(self) -> bool:
        return any("merge" in call and "--auto" in call for call in self.calls)


# --------------------------------------------------------------------------
# evaluate_hold
# --------------------------------------------------------------------------


def test_clean_pr_is_not_held() -> None:
    verdict = automerge_guard.evaluate_hold("o/r", 1, runner=FakeGh())
    assert verdict.held is False
    assert verdict.reasons == ()


@pytest.mark.parametrize(
    "label", ["do-not-merge", "blocked", "do-not-automate", "Do-Not-Merge"]
)
def test_hold_labels_block_arming(label: str) -> None:
    verdict = automerge_guard.evaluate_hold("o/r", 1, runner=FakeGh(labels=[label]))
    assert verdict.held is True
    assert f"`{label.lower()}` label" in verdict.describe()


def test_draft_is_held() -> None:
    verdict = automerge_guard.evaluate_hold("o/r", 1, runner=FakeGh(draft=True))
    assert verdict.held is True
    assert "draft" in verdict.describe()
    assert "mark ready first" in verdict.describe()


def test_human_disarm_after_head_commit_is_held() -> None:
    """The PR #4709 scenario: reviewer disarms, nobody pushes, automation re-arms."""
    verdict = automerge_guard.evaluate_hold(
        "o/r", 4709, runner=FakeGh(disarms=["2026-08-14T04:14:21Z"])
    )
    assert verdict.held is True
    assert "reviewer disabled auto-merge" in verdict.describe()


def test_human_disarm_before_head_commit_is_not_held() -> None:
    """A push after the disarm supersedes the reviewer's decision."""
    verdict = automerge_guard.evaluate_hold(
        "o/r", 1, runner=FakeGh(disarms=["2026-08-01T00:00:00Z"])
    )
    assert verdict.held is False


def test_future_dated_head_commit_cannot_outrun_a_disarm() -> None:
    """The committer date is contributor-controlled; arrival time is not."""
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(
            disarms=["2026-08-14T04:14:21Z"],
            head_date="2026-08-13T20:35:02Z",  # server time the SHA arrived
            committer_date="2099-01-01T00:00:00Z",  # forged
        ),
    )
    assert verdict.held is True


def test_force_push_to_a_previously_seen_sha_supersedes_a_disarm() -> None:
    """The SHA first got checks long ago; the force-push is when it became head."""
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(
            disarms=["2026-08-14T04:14:21Z"],
            head_date="2026-08-13T20:35:02Z",
            force_pushes=["2026-08-15T00:00:00Z"],
        ),
    )
    assert verdict.held is False


def test_force_push_before_the_disarm_still_holds() -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(
            disarms=["2026-08-14T04:14:21Z"],
            force_pushes=["2026-08-13T21:00:00Z"],
        ),
    )
    assert verdict.held is True


def test_timeline_failure_falls_back_to_check_suite_time() -> None:
    held = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(disarms=["2026-08-14T04:14:21Z"], force_push_fails=True),
    )
    assert held.held is True
    clear = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(disarms=["2026-08-01T00:00:00Z"], force_push_fails=True),
    )
    assert clear.held is False


def test_missing_arrival_time_fails_closed_when_a_disarm_exists() -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r", 1, runner=FakeGh(disarms=["2026-08-14T04:14:21Z"], head_date="")
    )
    assert verdict.held is True


def test_latest_disarm_wins_when_several_exist() -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(disarms=["2026-08-01T00:00:00Z", "2026-08-14T04:14:21Z"]),
    )
    assert verdict.held is True


def test_deleted_tracked_files_block_arming() -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r", 4709, runner=FakeGh(removed=[f"docs/f{i}.md" for i in range(13)])
    )
    assert verdict.held is True
    assert "deletes 13 tracked file(s)" in verdict.describe()
    assert "+8 more" in verdict.describe()


def test_deletions_released_by_label() -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(removed=["a.py"], labels=["deletions-acknowledged"]),
    )
    assert verdict.held is False


@pytest.mark.parametrize("value", ["yes", "true", "YES"])
def test_deletions_released_by_body_marker(value: str) -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(
            removed=["a.py"], body=f"why\n\nDeletions-Acknowledged: {value}\n"
        ),
    )
    assert verdict.held is False


def test_deletions_not_released_by_unrelated_body_text() -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        1,
        runner=FakeGh(removed=["a.py"], body="Deletions-Acknowledged: not yet"),
    )
    assert verdict.held is True


def test_multiple_signals_are_all_reported() -> None:
    verdict = automerge_guard.evaluate_hold(
        "o/r",
        4709,
        runner=FakeGh(removed=["a.py"], disarms=["2026-08-14T04:14:21Z"]),
    )
    assert len(verdict.reasons) == 2


def test_evaluation_failure_fails_closed() -> None:
    def broken(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _completed(returncode=1, stderr="HTTP 403 rate limited")

    verdict = automerge_guard.evaluate_hold("o/r", 1, runner=broken)
    assert verdict.held is True
    assert verdict.error is not None and "403" in verdict.error


# --------------------------------------------------------------------------
# arm_auto_merge
# --------------------------------------------------------------------------


def test_arm_refuses_held_pr_and_never_shells_out_to_merge() -> None:
    fake = FakeGh(labels=["do-not-merge"])
    result = automerge_guard.arm_auto_merge("o/r", 1, runner=fake)
    assert result.armed is False
    assert fake.armed is False, "a held PR must never reach `gh pr merge --auto`"


def test_arm_proceeds_on_clean_pr() -> None:
    fake = FakeGh()
    result = automerge_guard.arm_auto_merge("o/r", 1, runner=fake)
    assert result.armed is True
    assert fake.armed is True


def test_arm_passes_strategy_and_delete_branch() -> None:
    fake = FakeGh()
    automerge_guard.arm_auto_merge(
        "o/r", 7, strategy="rebase", delete_branch=True, runner=fake
    )
    merge_call = next(c for c in fake.calls if "merge" in c)
    assert "--rebase" in merge_call
    assert "--delete-branch" in merge_call
    assert merge_call[:5] == ["gh", "pr", "merge", "7", "--repo"]


def test_arm_reports_gh_failure() -> None:
    fake = FakeGh(merge_returncode=1)
    result = automerge_guard.arm_auto_merge("o/r", 1, runner=fake)
    assert result.armed is False
    assert "merge boom" in result.detail


GRAPHQL_BLOCKED = (
    "non-200 OK status code: 403 Forbidden body: "
    '"GitHub GraphQL is not available from Claude Code sessions; use the REST '
    'API ... PUT or DELETE /repos/{owner}/{repo}/pulls/{n}/ccr/auto_merge ..."'
)


class CloudGh(FakeGh):
    """FakeGh whose GraphQL arm fails, and which records the REST fallback."""

    def __init__(
        self,
        *,
        merge_stderr: str = GRAPHQL_BLOCKED,
        rest_returncode: int = 0,
        stored_method: str | None = None,
        verify_returncode: int = 0,
        delete_returncode: int = 0,
        put_body: str = "",
        **kw: object,
    ) -> None:
        super().__init__(merge_returncode=1, **kw)  # type: ignore[arg-type]
        self.put_body = put_body
        self.merge_stderr = merge_stderr
        self.rest_returncode = rest_returncode
        self.stored_method = stored_method
        self.verify_returncode = verify_returncode
        self.delete_returncode = delete_returncode

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(cmd)
        if ".auto_merge.merge_method" in argv:
            self.calls.append(argv)
            requested = next(
                (
                    p.split("=", 1)[1]
                    for c in self.rest_calls
                    for p in c
                    if p.startswith("merge_method=")
                ),
                "",
            )
            stored = self.stored_method if self.stored_method is not None else requested
            if self.verify_returncode:
                return _completed(returncode=self.verify_returncode, stderr="get boom")
            return _completed(stdout=stored + "\n")
        if any(part.endswith("/ccr/auto_merge") for part in argv):
            self.calls.append(argv)
            if "DELETE" in argv:
                return _completed(
                    returncode=self.delete_returncode,
                    stderr="delete boom" if self.delete_returncode else "",
                )
            return _completed(
                returncode=self.rest_returncode,
                stdout="" if self.rest_returncode else self.put_body,
                stderr="rest boom" if self.rest_returncode else "",
            )
        if "pr" in argv and "merge" in argv:
            self.calls.append(argv)
            return _completed(returncode=1, stderr=self.merge_stderr)
        return super().__call__(cmd)

    @property
    def rest_calls(self) -> list[list[str]]:
        return [c for c in self.calls if any(p.endswith("/ccr/auto_merge") for p in c)]


def test_graphql_blocked_falls_back_to_rest_route_once() -> None:
    fake = CloudGh()
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is True
    assert len(fake.rest_calls) == 1
    call = fake.rest_calls[0]
    assert call[:4] == ["gh", "api", "-X", "PUT"]
    assert "repos/o/r/pulls/7/ccr/auto_merge" in call
    assert "merge_method=squash" in call


def test_rest_fallback_uses_requested_strategy() -> None:
    fake = CloudGh()
    automerge_guard.arm_auto_merge("o/r", 7, strategy="rebase", runner=fake)
    assert "merge_method=rebase" in fake.rest_calls[0]


def test_rest_fallback_failure_is_reported_not_armed() -> None:
    fake = CloudGh(rest_returncode=1)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is False
    assert "rest boom" in result.detail
    assert len(fake.rest_calls) == 1


def test_rest_fallback_refuses_when_route_stores_a_different_method() -> None:
    """Codex P1 on #1937: never report armed for a strategy GitHub didn't store."""
    fake = CloudGh(stored_method="merge")
    result = automerge_guard.arm_auto_merge("o/r", 7, strategy="squash", runner=fake)
    assert result.armed is False
    assert "merge" in result.detail and "squash" in result.detail


def test_rest_fallback_verifies_the_stored_method() -> None:
    fake = CloudGh()
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is True
    assert any(".auto_merge.merge_method" in c for c in fake.calls)


def test_put_body_confirms_arm_even_when_pr_read_shows_it_queued() -> None:
    """#2001: arming a clean PR enqueues it, so the PR read shows auto_merge null.

    The PUT response reports what GitHub stored; trusting it must not revoke
    (and so dequeue) a correct arm.
    """
    body = '{"enabled":true,"merge_method":"squash"}'
    fake = CloudGh(put_body=body, stored_method="")
    result = automerge_guard.arm_auto_merge("o/r", 7, strategy="squash", runner=fake)
    assert result.armed is True
    assert not [c for c in fake.rest_calls if "DELETE" in c]


def test_put_body_with_a_different_method_is_still_revoked() -> None:
    fake = CloudGh(put_body='{"enabled":true,"merge_method":"merge"}')
    result = automerge_guard.arm_auto_merge("o/r", 7, strategy="squash", runner=fake)
    assert result.armed is False
    assert len([c for c in fake.rest_calls if "DELETE" in c]) == 1
    assert "'merge'" in result.detail


def test_put_body_reporting_disabled_is_revoked() -> None:
    fake = CloudGh(put_body='{"enabled":false,"merge_method":"squash"}')
    result = automerge_guard.arm_auto_merge("o/r", 7, strategy="squash", runner=fake)
    assert result.armed is False
    assert len([c for c in fake.rest_calls if "DELETE" in c]) == 1


def test_unparseable_put_body_falls_back_to_reading_the_pr() -> None:
    fake = CloudGh(put_body="not json")
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is True
    assert any(".auto_merge.merge_method" in c for c in fake.calls)


def _delete_calls(fake: CloudGh) -> list[list[str]]:
    return [c for c in fake.rest_calls if "DELETE" in c]


def test_verify_mismatch_revokes_the_arm() -> None:
    """Codex P1 on RD#1893: a PUT that stored the wrong method must be undone."""
    fake = CloudGh(stored_method="merge")
    result = automerge_guard.arm_auto_merge("o/r", 7, strategy="squash", runner=fake)
    assert result.armed is False
    deletes = _delete_calls(fake)
    assert len(deletes) == 1
    assert deletes[0][:4] == ["gh", "api", "-X", "DELETE"]
    assert "repos/o/r/pulls/7/ccr/auto_merge" in deletes[0]
    assert "revoked" in result.detail


@pytest.mark.parametrize("case", ["get-fails", "empty"])
def test_unverifiable_arm_is_revoked(case: str) -> None:
    fake = (
        CloudGh(verify_returncode=1)
        if case == "get-fails"
        else CloudGh(stored_method="")
    )
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is False
    assert len(_delete_calls(fake)) == 1


def test_failed_revoke_says_auto_merge_may_still_be_on() -> None:
    fake = CloudGh(stored_method="merge", delete_returncode=1)
    result = automerge_guard.arm_auto_merge("o/r", 7, strategy="squash", runner=fake)
    assert result.armed is False
    assert len(_delete_calls(fake)) == 1
    assert "may still be ON" in result.detail
    assert "manual" in result.detail.lower()
    assert "delete boom" in result.detail


def test_failed_revoke_flags_result_and_logs_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = CloudGh(stored_method="merge", delete_returncode=1)
    with caplog.at_level("ERROR"):
        result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.auto_merge_may_be_on is True
    assert any(
        r.levelname == "ERROR" and "may still be ON" in r.getMessage()
        for r in caplog.records
    )


def test_revoked_arm_does_not_flag_may_be_on() -> None:
    result = automerge_guard.arm_auto_merge(
        "o/r", 7, runner=CloudGh(stored_method="merge")
    )
    assert result.auto_merge_may_be_on is False


def test_cli_surfaces_detail_and_distinct_exit_when_revoke_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = automerge_guard.ArmResult(
        False,
        automerge_guard.HoldVerdict(False, ()),
        "revoke FAILED (x); auto-merge may still be ON and needs manual disarm",
        auto_merge_may_be_on=True,
    )
    monkeypatch.setattr(automerge_guard, "arm_auto_merge", lambda *a, **k: result)
    code = automerge_guard.main(["o/r", "1", "--arm"])
    out = capsys.readouterr().out
    assert code == automerge_guard.EXIT_ARM_MAY_BE_ON == 3
    assert "DANGER" in out and "may still be ON" in out


def test_cli_prints_detail_for_ordinary_unarmed_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    result = automerge_guard.ArmResult(
        False,
        automerge_guard.HoldVerdict(False, ()),
        "REST route stored merge method 'merge', not the requested 'squash'; "
        "auto-merge revoked",
    )
    monkeypatch.setattr(automerge_guard, "arm_auto_merge", lambda *a, **k: result)
    assert automerge_guard.main(["o/r", "1", "--arm"]) == 1
    assert "auto-merge revoked" in capsys.readouterr().out


def test_verified_arm_is_not_revoked() -> None:
    fake = CloudGh()
    assert automerge_guard.arm_auto_merge("o/r", 7, runner=fake).armed is True
    assert _delete_calls(fake) == []


def test_other_403_does_not_trigger_rest_fallback() -> None:
    fake = CloudGh(merge_stderr="403 Forbidden: Resource not accessible")
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is False
    assert fake.rest_calls == []


def test_held_pr_makes_no_graphql_or_rest_arm_call() -> None:
    fake = CloudGh(removed=["a.py"])
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is False
    assert not any("merge" in c and "pr" in c for c in fake.calls)
    assert fake.rest_calls == []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_reports_hold_with_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        automerge_guard,
        "evaluate_hold",
        lambda repo, pr, **kw: automerge_guard.HoldVerdict(True, ("`blocked` label",)),
    )
    assert automerge_guard.main(["o/r", "1"]) == 1
    assert "`blocked` label" in capsys.readouterr().out


def test_no_fleet_script_arms_auto_merge_directly() -> None:
    """Regression guard: every arm must route through automerge_guard.

    A direct `gh pr merge --auto` bypasses the reviewer's hold, which is the
    whole defect this module exists to close.
    """
    scripts = _MODULE_PATH.parent
    offenders: list[str] = []
    for path in sorted([*scripts.glob("*.py"), *scripts.glob("*.ps1")]):
        if path.name == _MODULE_PATH.name:
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith(("#", "//")):
                continue
            if "pr" in line and "merge" in line and "--auto" in line:
                offenders.append(f"{path.name}:{lineno}: {stripped}")
    assert not offenders, "direct auto-merge arming found:\n" + "\n".join(offenders)


def test_cli_reports_clear_with_zero_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        automerge_guard,
        "evaluate_hold",
        lambda repo, pr, **kw: automerge_guard.HoldVerdict(False),
    )
    assert automerge_guard.main(["o/r", "1"]) == 0
    assert "no hold in effect" in capsys.readouterr().out


# --- GOV-1 (#1917): warn when the active token is a personal/user token ---


@pytest.mark.parametrize(
    ("token", "kind"),
    [
        (FAKE_APP_TOKEN, "app"),
        (FAKE_CLASSIC_PAT, "user"),
        (FAKE_FINE_GRAINED_PAT, "user"),
        (FAKE_OAUTH_TOKEN, "user"),
        (FAKE_USER_TO_SERVER, "user"),
        ("v1.0123456789abcdef", "unknown"),
        ("", "none"),
        ("   ", "none"),
    ],
)
def test_classify_token(token: str, kind: str) -> None:
    assert automerge_guard.classify_token(token) == kind


def test_classify_token_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        automerge_guard.classify_token(None)  # type: ignore[arg-type]


def test_active_token_prefers_gh_token_then_github_token() -> None:
    def no_gh(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError("gh must not be consulted when an env token is set")

    env = {"GH_TOKEN": FAKE_CLASSIC_PAT, "GITHUB_TOKEN": FAKE_APP_TOKEN}
    assert automerge_guard.active_token_kind(env, runner=no_gh) == "user"
    env = {"GITHUB_TOKEN": FAKE_APP_TOKEN}
    assert automerge_guard.active_token_kind(env, runner=no_gh) == "app"


def test_active_token_falls_back_to_gh_auth_token() -> None:
    calls: list[list[str]] = []

    def gh(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return _completed(stdout=FAKE_OAUTH_TOKEN + "\n")

    assert automerge_guard.active_token_kind({}, runner=gh) == "user"
    assert calls == [["gh", "auth", "token"]]


def test_active_token_is_none_when_gh_is_unauthenticated_or_missing() -> None:
    def unauth(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _completed(returncode=1, stderr="no oauth token")

    def missing(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gh")

    assert automerge_guard.active_token_kind({}, runner=unauth) == "none"
    assert automerge_guard.active_token_kind({}, runner=missing) == "none"


def test_identity_check_warns_on_user_token_without_leaking_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = {"GH_TOKEN": FAKE_CLASSIC_PAT}
    with caplog.at_level("WARNING"):
        ok = automerge_guard.check_token_identity(env=env, require_app_token=False)
    assert ok is True
    assert "personal/user token" in caplog.text
    assert FAKE_CLASSIC_PAT not in caplog.text
    assert "B" * 20 not in caplog.text


def test_identity_check_is_silent_for_app_token(
    caplog: pytest.LogCaptureFixture,
) -> None:
    env = {"GH_TOKEN": FAKE_APP_TOKEN}
    with caplog.at_level("WARNING"):
        assert automerge_guard.check_token_identity(env=env, require_app_token=True)
    assert caplog.text == ""


@pytest.mark.parametrize("token", [FAKE_FINE_GRAINED_PAT, "", "v1.opaque"])
def test_identity_check_fails_only_when_app_token_required(token: str) -> None:
    env = {"GH_TOKEN": token} if token else {}

    def unauth(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _completed(returncode=1)

    assert automerge_guard.check_token_identity(
        env=env, require_app_token=False, runner=unauth
    )
    assert not automerge_guard.check_token_identity(
        env=env, require_app_token=True, runner=unauth
    )


def test_cli_require_app_token_refuses_user_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GH_TOKEN", FAKE_CLASSIC_PAT)

    def must_not_evaluate(repo: str, pr: int, **kw: object) -> object:
        raise AssertionError("must refuse before touching the PR")

    monkeypatch.setattr(automerge_guard, "evaluate_hold", must_not_evaluate)
    assert automerge_guard.main(["o/r", "1", "--require-app-token"]) == 2
    captured = capsys.readouterr()
    assert FAKE_CLASSIC_PAT not in captured.out + captured.err


def test_cli_user_token_without_flag_warns_but_proceeds(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("GH_TOKEN", FAKE_CLASSIC_PAT)
    monkeypatch.setattr(
        automerge_guard,
        "evaluate_hold",
        lambda repo, pr, **kw: automerge_guard.HoldVerdict(False),
    )
    with caplog.at_level("WARNING"):
        assert automerge_guard.main(["o/r", "1"]) == 0
    # main() logs through logging.basicConfig, whose handler writes to stderr.
    assert "personal/user token" in caplog.text
    captured = capsys.readouterr()
    assert FAKE_CLASSIC_PAT not in captured.out + captured.err + caplog.text


# --------------------------------------------------------------------------
# #1968: arming refused because the PR is already clean -> merge directly
# --------------------------------------------------------------------------

CLEAN_STATUS_422 = (
    "gh: Pull request is in clean status, so auto-merge cannot be enabled (HTTP 422)"
)


class CleanGh(CloudGh):
    """CloudGh whose arm PUT is refused, and which serves the direct merge."""

    def __init__(
        self,
        *,
        arm_stderr: str = CLEAN_STATUS_422,
        direct_returncode: int = 0,
        direct_stderr: str = "",
        merged: str = "true",
        queue: bool = False,
        rules_fail: bool = False,
        graphql_fail: bool = False,
        queue_on_later_page: bool = False,
        later_page_fails: bool = False,
        ref_delete_returncode: int = 0,
        **kw: object,
    ) -> None:
        super().__init__(rest_returncode=1, **kw)
        self.queue_on_later_page = queue_on_later_page
        self.later_page_fails = later_page_fails
        self.ref_delete_returncode = ref_delete_returncode
        self.queue = queue
        self.rules_fail = rules_fail
        self.graphql_fail = graphql_fail
        self.arm_stderr = arm_stderr
        self.direct_returncode = direct_returncode
        self.direct_stderr = direct_stderr
        self.merged = merged

    def __call__(self, cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(cmd)
        if any(p.endswith("/ccr/auto_merge") for p in argv) and "PUT" in argv:
            self.calls.append(argv)
            return _completed(returncode=1, stderr=self.arm_stderr)
        if any("/rules/branches/main" in p for p in argv):
            self.calls.append(argv)
            if self.rules_fail:
                return _completed(returncode=1, stderr="rules boom")
            # Like real gh: --paginate + --jq flattens every page to lines.
            paged = "--paginate" in argv and "per_page=100" in " ".join(argv)
            types = ["deletion"]
            if self.queue:
                types.append("merge_queue")
            if self.queue_on_later_page:
                if not paged:
                    return _completed(stdout=json.dumps([{"type": "deletion"}]))
                if self.later_page_fails:
                    return _completed(returncode=1, stderr="page 2 boom")
                types.append("merge_queue")
            return _completed(stdout="\n".join(types) + "\n")
        if any("/git/refs/heads/" in p for p in argv):
            self.calls.append(argv)
            return _completed(
                returncode=self.ref_delete_returncode,
                stderr="ref boom" if self.ref_delete_returncode else "",
            )
        if "graphql" in argv:
            self.calls.append(argv)
            return _completed(
                returncode=1 if self.graphql_fail else 0,
                stderr=GRAPHQL_BLOCKED if self.graphql_fail else "",
                stdout="" if self.graphql_fail else "{}",
            )
        if any(p.endswith("/pulls/7/merge") for p in argv):
            self.calls.append(argv)
            return _completed(
                returncode=self.direct_returncode, stderr=self.direct_stderr
            )
        if "--jq" in argv and ".merged" in argv:
            self.calls.append(argv)
            return _completed(stdout=self.merged + "\n")
        return super().__call__(cmd)

    @property
    def direct_calls(self) -> list[list[str]]:
        return [c for c in self.calls if any(p.endswith("/pulls/7/merge") for p in c)]


def test_clean_status_422_merges_directly_with_sha_and_method() -> None:
    fake = CleanGh()
    result = automerge_guard.arm_auto_merge("o/r", 7, strategy="rebase", runner=fake)
    assert result.armed is False
    assert result.merged is True
    assert "already clean" in result.detail
    assert len(fake.direct_calls) == 1
    call = fake.direct_calls[0]
    assert call[:4] == ["gh", "api", "-X", "PUT"]
    assert "merge_method=rebase" in call
    assert f"sha={HEAD_SHA}" in call
    assert not any("admin" in part for part in call)


def test_clean_status_merge_is_verified_via_merged_flag() -> None:
    fake = CleanGh(merged="false")
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is False
    assert "merged" in result.reason() and "not" in result.reason()
    assert len(fake.direct_calls) == 1


@pytest.mark.parametrize("status", ["405", "409", "422"])
def test_clean_status_merge_refused_reports_reason_without_retry(status: str) -> None:
    fake = CleanGh(
        direct_returncode=1, direct_stderr=f"Head branch was modified (HTTP {status})"
    )
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is False
    assert result.armed is False
    assert "Head branch was modified" in result.detail
    assert len(fake.direct_calls) == 1
    assert len(fake.rest_calls) == 1


def test_other_422_keeps_current_behaviour() -> None:
    fake = CleanGh(arm_stderr="Auto merge is not allowed for this repository (422)")
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is False
    assert result.armed is False
    assert "not allowed" in result.detail
    assert fake.direct_calls == []


def test_held_pr_is_never_merged_directly() -> None:
    fake = CleanGh(removed=["a.py"])
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is False
    assert fake.direct_calls == []


ENQUEUE_HINT = "gh pr merge 7 --repo o/r --squash"


def _graphql_calls(fake: CleanGh) -> list[list[str]]:
    return [c for c in fake.calls if "graphql" in c]


def test_queue_present_enqueues_with_expected_head_and_never_puts_merge() -> None:
    fake = CleanGh(queue=True)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is True
    assert result.merged is False
    assert "enqueued" in result.detail
    gql = _graphql_calls(fake)
    assert len(gql) == 1
    text = " ".join(gql[0])
    assert "enqueuePullRequest" in text and "expectedHeadOid" in text
    assert f"oid={HEAD_SHA}" in gql[0] and "id=PR_node7" in gql[0]
    assert fake.direct_calls == []


def test_queue_present_graphql_failure_never_direct_merges() -> None:
    fake = CleanGh(queue=True, graphql_fail=True)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.armed is False and result.merged is False
    assert ENQUEUE_HINT in result.detail
    assert fake.direct_calls == []
    assert len(_graphql_calls(fake)) == 1


def test_rules_lookup_failure_is_treated_as_queue_present() -> None:
    fake = CleanGh(rules_fail=True)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is False
    assert fake.direct_calls == []
    assert len(_graphql_calls(fake)) == 1


def test_no_queue_uses_direct_merge_path() -> None:
    fake = CleanGh(queue=False)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is True
    assert _graphql_calls(fake) == []


def test_enqueue_hint_names_the_repo() -> None:
    fake = CleanGh(queue=True, graphql_fail=True)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert "gh pr merge 7 --repo o/r --squash" in result.detail


def test_rules_lookup_is_paginated_at_100() -> None:
    fake = CleanGh(queue_on_later_page=True)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is False, "a queue on page 2 must not be missed"
    assert fake.direct_calls == []
    assert len(_graphql_calls(fake)) == 1


def test_rules_later_page_failure_fails_closed() -> None:
    fake = CleanGh(queue_on_later_page=True, later_page_fails=True)
    result = automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert result.merged is False
    assert fake.direct_calls == []


def _ref_deletes(fake: CleanGh) -> list[list[str]]:
    return [c for c in fake.calls if any("/git/refs/heads/" in p for p in c)]


def test_clean_fallback_honours_delete_branch() -> None:
    fake = CleanGh()
    result = automerge_guard.arm_auto_merge("o/r", 7, delete_branch=True, runner=fake)
    assert result.merged is True
    deletes = _ref_deletes(fake)
    assert len(deletes) == 1
    assert deletes[0] == [
        "gh",
        "api",
        "-X",
        "DELETE",
        "repos/o/r/git/refs/heads/feat/x",
    ]


def test_clean_fallback_percent_encodes_the_head_ref() -> None:
    # `gh api` treats `#` as a URL fragment: an unencoded `feat#one` would send
    # DELETE .../heads/feat and delete an unrelated branch. `/` stays literal.
    fake = CleanGh()
    fake.pull["head"]["ref"] = "feat/x#one?y%z"  # type: ignore[index]
    automerge_guard.arm_auto_merge("o/r", 7, delete_branch=True, runner=fake)
    deletes = _ref_deletes(fake)
    assert len(deletes) == 1
    assert deletes[0][-1] == "repos/o/r/git/refs/heads/feat/x%23one%3Fy%25z"


def test_clean_fallback_does_not_delete_without_the_flag() -> None:
    fake = CleanGh()
    automerge_guard.arm_auto_merge("o/r", 7, runner=fake)
    assert _ref_deletes(fake) == []


def test_clean_fallback_never_deletes_a_fork_branch() -> None:
    fake = CleanGh()
    fake.pull["head"]["repo"] = {"full_name": "fork/r"}  # type: ignore[index]
    result = automerge_guard.arm_auto_merge("o/r", 7, delete_branch=True, runner=fake)
    assert result.merged is True
    assert _ref_deletes(fake) == []


def test_clean_fallback_delete_failure_warns_but_stays_merged() -> None:
    fake = CleanGh(ref_delete_returncode=1)
    result = automerge_guard.arm_auto_merge("o/r", 7, delete_branch=True, runner=fake)
    assert result.merged is True
    assert "branch not deleted" in result.detail


def test_clean_fallback_does_not_delete_when_merge_unverified() -> None:
    fake = CleanGh(merged="false")
    result = automerge_guard.arm_auto_merge("o/r", 7, delete_branch=True, runner=fake)
    assert result.merged is False
    assert _ref_deletes(fake) == []


def test_cli_reports_merged_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        automerge_guard, "arm_auto_merge", _patched_merged_result(merged=True)
    )
    assert automerge_guard.main(["o/r", "7", "--arm"]) == 0
    assert "merged o/r#7 (squash; PR was already clean)" in capsys.readouterr().out


def test_cli_refused_clean_merge_exits_nonzero_with_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        automerge_guard, "arm_auto_merge", _patched_merged_result(merged=False)
    )
    assert automerge_guard.main(["o/r", "7", "--arm"]) == 1
    assert "merge queue required" in capsys.readouterr().out


def _patched_merged_result(*, merged: bool):  # type: ignore[no-untyped-def]
    result = automerge_guard.ArmResult(
        False,
        automerge_guard.HoldVerdict(False, ()),
        ""
        if merged
        else "PR was already clean but direct merge refused: merge queue required",
        merged=merged,
    )
    return lambda *a, **k: result


# --------------------------------------------------------------------------
# enqueue_stalled_pr (#2018): re-enqueue a green, armed PR the queue missed
# --------------------------------------------------------------------------


def test_stalled_pr_on_queue_base_is_enqueued_once_at_the_verified_head() -> None:
    fake = CleanGh(queue=True)
    result = automerge_guard.enqueue_stalled_pr("o/r", 7, runner=fake)
    assert result.armed is True
    assert "enqueued" in result.detail
    gql = _graphql_calls(fake)
    assert len(gql) == 1
    assert f"oid={HEAD_SHA}" in gql[0] and "id=PR_node7" in gql[0]
    assert fake.direct_calls == []
    assert not fake.armed  # no `gh pr merge --auto` round trip


def test_stalled_pr_that_is_held_is_never_enqueued() -> None:
    fake = CleanGh(queue=True, labels=["do-not-merge"])
    result = automerge_guard.enqueue_stalled_pr("o/r", 7, runner=fake)
    assert result.armed is False
    assert result.detail == "held"
    assert _graphql_calls(fake) == []
    assert fake.direct_calls == []


def test_stalled_pr_without_a_merge_queue_is_left_alone() -> None:
    fake = CleanGh(queue=False)
    result = automerge_guard.enqueue_stalled_pr("o/r", 7, runner=fake)
    assert result.armed is False and result.merged is False
    assert "no merge queue" in result.detail
    assert _graphql_calls(fake) == []
    assert fake.direct_calls == []


def test_stalled_pr_enqueue_failure_is_reported_not_retried() -> None:
    fake = CleanGh(queue=True, graphql_fail=True)
    result = automerge_guard.enqueue_stalled_pr("o/r", 7, runner=fake)
    assert result.armed is False
    assert len(_graphql_calls(fake)) == 1
    assert fake.direct_calls == []
