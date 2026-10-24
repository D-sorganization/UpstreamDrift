"""actionlint gate for GitHub workflow expression validity (UD-CI-031 / #13572)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
RUN_ACTIONLINT = REPO_ROOT / "scripts" / "ci" / "run_actionlint.py"
BROKEN_FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "ci"
    / "cross_engine_leaderboard_publish_broken_if.yml"
)
PUBLISH_WORKFLOW = WORKFLOWS / "cross-engine-leaderboard-publish.yml"
HISTORICAL_ACTIONLINT_SNIPPET = "unexpected end of input while parsing closing ')'"


def _load_run_actionlint():
    spec = importlib.util.spec_from_file_location(
        "run_actionlint_under_test", RUN_ACTIONLINT
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def actionlint_runner():
    return _load_run_actionlint()


def test_cross_engine_leaderboard_publish_workflow_passes_actionlint(
    actionlint_runner,
) -> None:
    result = actionlint_runner.run_actionlint([PUBLISH_WORKFLOW])
    assert result.returncode == 0, result.stderr or result.stdout


def test_regression_unclosed_job_if_expression_fails_actionlint(
    actionlint_runner,
) -> None:
    """Matches the Oct 2026 fleet failure: job `if` missing a closing parenthesis."""
    result = actionlint_runner.run_actionlint([BROKEN_FIXTURE])
    assert result.returncode != 0
    combined = (result.stdout or "") + (result.stderr or "")
    assert HISTORICAL_ACTIONLINT_SNIPPET in combined


def test_lint_workflow_files_runs_actionlint_on_changed_paths() -> None:
    lint = (WORKFLOWS / "lint-workflow-files.yml").read_text(encoding="utf-8")
    assert "Run actionlint on modified workflows" in lint
    assert "download-actionlint.bash" in lint
    assert 'actionlint" -config-file .github/actionlint.yaml' in lint
    assert "rm -f actionlint" in lint
