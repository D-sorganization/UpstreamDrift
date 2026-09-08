"""Fail-safe SPEC.md reminder contracts for the Spec Check workflow (issue #9499).

The `Verify SPEC.md freshness` job used to embed its comment-posting logic as an
inline `actions/github-script` heredoc. An unescaped backtick inside that
heredoc aborted the step with `SyntaxError: Invalid or unexpected token`, so the
diagnostic never reached the contributor. These tests pin the extraction of the
posting logic into a testable script plus its fail-safe contract: a posting
failure must never swallow the finding — the diagnostic must land in the job
log and the job must still exit non-zero.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import post_spec_reminder

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/spec-check.yml"
pytestmark = pytest.mark.unit


def _workflow_steps() -> list[dict[str, Any]]:
    """Parse the Spec Check workflow and return the freshness job's steps."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return list(workflow["jobs"]["spec-freshness"]["steps"])


def _step_by_name(name: str) -> dict[str, Any]:
    steps = _workflow_steps()
    return next(step for step in steps if step.get("name") == name)


class TestWorkflowDelegatesPostingToTestableScript:
    """The comment body must live in a reviewed script, not a YAML heredoc."""

    def test_posting_step_runs_the_extracted_script(self) -> None:
        step = _step_by_name("Post warning comment")
        assert "uses" not in step, "posting must not be an inline github-script"
        run = step.get("run", "")
        assert "scripts/post_spec_reminder.py" in run

    def test_no_github_script_heredoc_remains(self) -> None:
        for step in _workflow_steps():
            script = (step.get("with") or {}).get("script", "")
            assert "SPEC.md Update Required" not in script

    def test_fail_step_runs_even_if_posting_step_fails(self) -> None:
        fail_step = _step_by_name("Fail if spec is stale")
        condition = fail_step.get("if", "")
        assert "always()" in condition
        assert "steps.check.outputs.needs_update == 'true'" in condition


class TestReminderScriptFailurePath:
    """A reporting failure must log the diagnostic and keep the gate armed."""

    def test_posting_failure_prints_diagnostic_and_returns_zero(self) -> None:
        def broken_request(method: str, path: str, token: str, payload: Any = None):
            raise OSError("network unreachable")

        original = post_spec_reminder._request
        post_spec_reminder._request = broken_request
        try:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = post_spec_reminder.main(
                    [
                        "--repository",
                        "D-sorganization/UpstreamDrift",
                        "--issue-number",
                        "42",
                    ]
                )
        finally:
            post_spec_reminder._request = original

        assert code == 0
        output = buffer.getvalue()
        assert "SPEC.md Update Required" in output, "diagnostic missing from job log"
        assert "spec-exempt" in output, "resolution guidance missing from job log"

    def test_missing_token_prints_diagnostic_and_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = post_spec_reminder.main(
                [
                    "--repository",
                    "D-sorganization/UpstreamDrift",
                    "--issue-number",
                    "42",
                ]
            )

        assert code == 0
        assert "SPEC.md Update Required" in buffer.getvalue()


class TestReminderScriptSuccessPath:
    """Happy path: post once, then suppress the duplicate."""

    def _fake_request(self, calls: list[tuple[str, str, Any]]):
        def request(method: str, path: str, token: str, payload: Any = None):
            calls.append((method, path, payload))
            if path.endswith("/comments"):
                return {"data": []} if method == "GET" else {"data": {"id": 1}}
            raise AssertionError(f"unexpected API path {path}")

        return request

    def test_posts_comment_when_none_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, Any]] = []
        monkeypatch.setattr(post_spec_reminder, "_request", self._fake_request(calls))
        monkeypatch.setenv("GITHUB_TOKEN", "t")

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = post_spec_reminder.main(
                [
                    "--repository",
                    "D-sorganization/UpstreamDrift",
                    "--issue-number",
                    "42",
                ]
            )

        assert code == 0
        methods = [method for method, _, _ in calls]
        assert methods == ["GET", "POST"]
        body = calls[1][2]["body"]
        assert "SPEC.md Update Required" in body

    def test_skips_duplicate_comment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, Any]] = []

        def request(method: str, path: str, token: str, payload: Any = None):
            calls.append((method, path, payload))
            return {
                "data": [
                    {
                        "body": "## -- SPEC.md Update Required",
                        "user": {"type": "Bot"},
                    }
                ]
            }

        monkeypatch.setattr(post_spec_reminder, "_request", request)
        monkeypatch.setenv("GITHUB_TOKEN", "t")

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = post_spec_reminder.main(
                [
                    "--repository",
                    "D-sorganization/UpstreamDrift",
                    "--issue-number",
                    "42",
                ]
            )

        assert code == 0
        assert [method for method, _, _ in calls] == ["GET"]


class TestReminderScriptInputValidation:
    @pytest.mark.parametrize(
        "argv",
        [
            ["--repository", "no-slash", "--issue-number", "42"],
            ["--repository", "D-sorganization/UpstreamDrift", "--issue-number", "0"],
            ["--repository", "D-sorganization/UpstreamDrift", "--issue-number", "-3"],
        ],
    )
    def test_rejects_malformed_input(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "t")
        with pytest.raises(ValueError):
            post_spec_reminder.main(argv)


class TestReminderBodyContent:
    """The body carries the actionable guidance from the policy."""

    def test_body_names_policy_link_and_change_log_row_format(self) -> None:
        body = post_spec_reminder.build_reminder_body(
            post_spec_reminder.FALLBACK_POLICY_URL
        )
        assert "SPECIFICATION_DRIVEN_DEVELOPMENT_POLICY.md" in body
        assert "| YYYY-MM-DD | #<pr> | summary |" in body
        assert "spec-exempt" in body
