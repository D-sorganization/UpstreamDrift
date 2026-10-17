"""Workflow steps that pipe through ``tee`` must propagate failures (#12963).

GitHub's implicit bash is ``bash -e {0}`` which does NOT set ``pipefail``, so
``pytest ... | tee log`` exits 0 even when pytest fails. A step is safe only
when it runs under an explicit ``shell: bash`` (``-eo pipefail``) or sets
pipefail itself.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
_TEE = re.compile(r"\|\s*tee\b")
_PIPEFAIL = re.compile(r"pipefail")
_NON_BASH = ("pwsh", "powershell", "cmd", "python")


def _shell_is_safe(shell: Any) -> bool | None:
    """True/False if a shell is declared; None if undeclared (implicit bash)."""
    if not shell:
        return None
    text = str(shell).strip()
    if text.split()[0] in _NON_BASH:
        return True  # not a bash pipeline; out of scope
    if text == "bash":
        return True  # explicit bash => --noprofile --norc -eo pipefail {0}
    return bool(_PIPEFAIL.search(text))


def _default_shell(node: Any) -> Any:
    return ((node or {}).get("defaults") or {}).get("run", {}).get("shell")


def find_unsafe_tee_steps(data: dict[str, Any]) -> list[str]:
    """Return ``job/step`` names whose ``| tee`` lacks pipefail semantics."""
    unsafe: list[str] = []
    for job_id, job in (data.get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            run = step.get("run")
            if not isinstance(run, str) or not _TEE.search(run):
                continue
            shell = step.get("shell") or _default_shell(job) or _default_shell(data)
            declared = _shell_is_safe(shell)
            if declared is None:
                declared = False
            if declared or _PIPEFAIL.search(run):
                continue
            unsafe.append(f"{job_id}/{step.get('name', index)}")
    return unsafe


def _workflow_files() -> list[Path]:
    return sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])


def test_checker_flags_unsafe_and_accepts_safe_steps() -> None:
    bad = {"jobs": {"j": {"steps": [{"name": "s", "run": "pytest | tee a.log"}]}}}
    assert find_unsafe_tee_steps(bad) == ["j/s"]
    legacy = {"jobs": {"j": {"steps": [{"run": "x | tee a", "shell": "bash -e {0}"}]}}}
    assert find_unsafe_tee_steps(legacy) == ["j/0"]
    for ok in (
        {"run": "x | tee a", "shell": "bash"},
        {"run": "set -o pipefail\nx | tee a"},
    ):
        assert find_unsafe_tee_steps({"jobs": {"j": {"steps": [ok]}}}) == []
    defaulted = {
        "defaults": {"run": {"shell": "bash"}},
        "jobs": {"j": {"steps": [{"run": "x | tee a"}]}},
    }
    assert find_unsafe_tee_steps(defaulted) == []


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_workflow_tee_steps_propagate_failures(path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert find_unsafe_tee_steps(data) == [], (
        f"{path.name}: steps pipe through tee without pipefail; "
        "add `shell: bash` or `set -o pipefail`"
    )
