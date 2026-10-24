"""Hermetic execution tests for the collation PR update preflight."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, cast

import yaml
import pytest

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "collate-changes.yml"
)
PROPOSAL_STEP = "Propose collation as a pull request"


def _proposal_script() -> str:
    data = cast("dict[str, Any]", yaml.safe_load(WORKFLOW.read_text("utf-8")))
    steps = data["jobs"]["collate"]["steps"]
    step = next(item for item in steps if item.get("name") == PROPOSAL_STEP)
    return cast("str", step["run"])


def _run_proposal(
    tmp_path: Path, helper_status: int | None, *, unavailable: bool = False
) -> tuple[int, str, list[str]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    operations = tmp_path / "operations"
    (bin_dir / "git").write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$OP_LOG"\n'
        'case "$1" in\n'
        '  status) echo " M SPEC.md" ;;\n'
        "  config|checkout|add|commit|push) exit 0 ;;\n"
        '  *) echo "unexpected fake git command: $*" >&2; exit 97 ;;\n'
        "esac\n"
    )
    (bin_dir / "gh").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        '  *"pulls?state=open&head=o:bot/collate-changes"*) echo 7 ;;\n'
        '  *) echo "unexpected fake gh command: $*" >&2; exit 97 ;;\n'
        "esac\n"
    )
    helper_case = '".rm-tools/scripts/pr_update_guard.py") exit 97 ;;'
    if not unavailable:
        helper_code = "exit 97" if helper_status is None else f"exit {helper_status}"
        helper_case = f'".rm-tools/scripts/pr_update_guard.py") {helper_code} ;;'
    (bin_dir / "python3").write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  {helper_case}\n"
        '  ".rm-tools/scripts/automerge_guard.py")\n'
        '    case " $* " in *" --arm "*) printf "arm\\n" >> "$OP_LOG" ;; esac\n'
        "    exit 0 ;;\n"
        '  *) echo "unexpected fake python invocation: $*" >&2; exit 97 ;;\n'
        "esac\n"
    )
    for executable in bin_dir.iterdir():
        executable.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "OP_LOG": str(operations),
            "GITHUB_ENV": str(tmp_path / "github-env"),
            "REPO_SLUG": "o/r",
            "BRANCH": "bot/collate-changes",
            "COLLATED": "1",
            "GUARD": ".rm-tools/scripts/automerge_guard.py",
            "UPDATE_GUARD": ".rm-tools/scripts/pr_update_guard.py",
        }
    )
    result = subprocess.run(
        ["bash", "-c", _proposal_script()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    recorded = operations.read_text().splitlines() if operations.exists() else []
    return result.returncode, result.stdout + result.stderr, recorded


def test_queued_pr_defers_before_checkout_commit_push_or_rearm(tmp_path: Path) -> None:
    """A queued PR must not be overwritten, even without a reviewer hold."""
    code, output, operations = _run_proposal(tmp_path, helper_status=1)
    assert code == 0, output
    assert not any(
        item.startswith(("checkout ", "commit ", "push ", "arm")) for item in operations
    ), operations
    assert "not editable" in output.lower() or "deferred" in output.lower()


def test_held_pr_defers_before_any_branch_mutation(tmp_path: Path) -> None:
    code, output, operations = _run_proposal(tmp_path, helper_status=2)
    assert code == 0, output
    assert not any(
        item.startswith(("checkout ", "commit ", "push ", "arm")) for item in operations
    )


def test_unavailable_helper_fails_closed_before_branch_mutation(tmp_path: Path) -> None:
    code, output, operations = _run_proposal(
        tmp_path, helper_status=None, unavailable=True
    )
    assert code != 0 or "deferred" in output.lower() or "not editable" in output.lower()
    assert not any(
        item.startswith(("checkout ", "commit ", "push ", "arm")) for item in operations
    )


def test_editable_pr_keeps_the_normal_push_and_final_arm_path(tmp_path: Path) -> None:
    code, output, operations = _run_proposal(tmp_path, helper_status=0)
    assert code == 0, output
    assert any(item.startswith("checkout ") for item in operations)
    assert any(item.startswith("push ") for item in operations)
    assert "arm" in operations


def test_both_guards_use_the_canonical_runtime_checkout() -> None:
    data = yaml.safe_load(WORKFLOW.read_text("utf-8"))
    steps = data["jobs"]["collate"]["steps"]
    proposal = next(step for step in steps if step.get("name") == PROPOSAL_STEP)
    assert proposal["env"]["GUARD"] == ".rm-tools/scripts/automerge_guard.py"
    assert proposal["env"]["UPDATE_GUARD"] == ".rm-tools/scripts/pr_update_guard.py"
    assert 'python3 "$UPDATE_GUARD" "$REPO_SLUG" "$PR_NUM"' in proposal["run"]
    assert (
        'python3 "$GUARD" "$REPO_SLUG" "$PR_NUM" --arm --strategy squash'
        in proposal["run"]
    )
