"""Runtime sentinel for issue #9387 (worker corruption, victims rotate).

Runs each documented #9387 victim file in its own serial subprocess
(``-p no:xdist``, one file per ``pytest.main`` call) and asserts that the
``src`` package identity and every ``src.*`` entry in ``sys.modules`` are
unchanged across the run.  A leaking pivot test (``sys.modules["src"] =
...``, or an un-restored ``pop``/``del`` of a ``src.*`` entry followed by
a re-import) changes the recorded identity map and fails the sentinel,
even though it corrupts only later tests scheduled to the same worker
rather than itself.

Complements the static conftest tripwire added by sibling PR #9723
(``tests/unit/repo_hygiene/test_no_conftest_src_module_pivot.py``):
that one inspects conftest sources statically; this one exercises the
victim files from the issue at runtime.

Environment tolerance (documented, not hidden):

- In a ``git worktree`` checkout ``.git`` is a *file*, so two
  ``tests/scripts/test_validate_suite.py`` tests that read
  ``.git/config``/``.git/HEAD`` fail for reasons unrelated to module
  state.  Those two ids are tolerated as environment-dependent failures
  only when ``.git`` is a file; in a normal checkout (and in CI) they
  must pass.
- If a victim file cannot even be collected because an optional heavy
  dependency is missing on this platform, that file is reported as
  skipped with the missing module named instead of failing the
  sentinel.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

# The four victim files documented on issue #9387 (the issue body's
# ``tests/unit/api/utils`` path is wrong; the real file is
# ``tests/unit/utils/test_path_validation.py``).
VICTIM_FILES: tuple[str, ...] = (
    "tests/launchers/test_shot_tracer_import_gui.py",
    "tests/scripts/test_validate_suite.py",
    "tests/unit/test_unreal_integration/test_mesh_loader.py",
    "tests/unit/utils/test_path_validation.py",
)

# validate_suite test ids that fail in a worktree checkout (``.git`` is a
# gitdir pointer file there) for reasons unrelated to sys.modules state.
_WORKTREE_ENV_FAILURES: dict[str, set[tuple[str, str]]] = {
    "tests/scripts/test_validate_suite.py": {
        (
            "tests.scripts.test_validate_suite",
            "test_git_repository_check_uses_the_project_root",
        ),
        (
            "tests.scripts.test_validate_suite",
            "test_comprehensive_validation_passes_end_to_end",
        ),
    },
}

_RUNNER_SOURCE = '''\
"""Sentinel runner: snapshot sys.modules around one pytest.main run."""
import contextlib
import io
import json
import os
import sys
import xml.etree.ElementTree as ET


def _snapshot() -> dict:
    """Record src-package identity and every src.* module object id."""
    modules = {
        name: [id(mod), getattr(mod, "__file__", None)]
        for name, mod in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    src = sys.modules.get("src")
    return {"src_id": id(src) if src is not None else None, "modules": modules}


def _outcomes(junit_path: str) -> list:
    """Flatten a junitxml report into (classname, name, outcome) triples."""
    cases = []
    for case in ET.parse(junit_path).getroot().iter("testcase"):
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error"):
                outcome = child.tag
            elif child.tag == "skipped":
                outcome = "skipped"
        cases.append((case.get("classname", ""), case.get("name", ""), outcome))
    return cases


sys.path.insert(0, os.getcwd())
import src  # noqa: E402  (establish the canonical identity baseline)
import pytest  # noqa: E402

victim, junit_path = sys.argv[1], sys.argv[2]
before = _snapshot()
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exit_code = pytest.main(
        [
            "--confcutdir=tests",
            "-p",
            "no:xdist",
            "-q",
            f"--junitxml={junit_path}",
            victim,
        ]
    )
after = _snapshot()
print(
    "SENTINEL_JSON:"
    + json.dumps(
        {
            "exit": int(exit_code),
            "before": before,
            "after": after,
            "outcomes": _outcomes(junit_path),
            "tail": captured.getvalue()[-4000:],
        }
    )
)
'''


def _run_victim(tmp_path: Path, victim: str) -> dict[str, Any]:
    """Run one victim file in a sentinel subprocess and parse its report."""
    runner = tmp_path / "sentinel_runner.py"
    junit = tmp_path / f"{victim.replace('/', '_').replace('.', '_')}.xml"
    runner.write_text(_RUNNER_SOURCE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(runner), victim, str(junit)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    markers = [ln for ln in proc.stdout.splitlines() if ln.startswith("SENTINEL_JSON:")]
    if not markers:
        pytest.fail(
            f"sentinel runner produced no report for {victim}\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    report = json.loads(markers[-1].removeprefix("SENTINEL_JSON:"))
    report["returncode"] = proc.returncode
    return report


def _assert_namespace_unchanged(report: dict[str, Any], victim: str) -> None:
    """Fail if a src.* entry was re-bound, dropped, or pivoted by the run."""
    before, after = report["before"], report["after"]
    if before["src_id"] is not None and before["src_id"] != after["src_id"]:
        pytest.fail(
            f"{victim} replaced sys.modules['src'] with a different module "
            f"object ({before['src_id']} -> {after['src_id']}); this is the "
            "#9387 worker-corruption signature"
        )
    lost = [n for n in before["modules"] if n not in after["modules"]]
    re_bound = [
        n
        for n, entry in before["modules"].items()
        if n in after["modules"] and after["modules"][n] != entry
    ]
    if lost or re_bound:
        pytest.fail(
            f"{victim} mutated src.* sys.modules entries (#9387 signature); "
            f"dropped: {sorted(lost)}; re-bound to new objects: {sorted(re_bound)}"
        )


def _missing_dependency(report: dict[str, Any]) -> str | None:
    """Return the missing module if collection died on an import error."""
    match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", report["tail"])
    if match and report["exit"] != 0:
        return match.group(1)
    return None


@pytest.mark.unit
@pytest.mark.timeout(600)
@pytest.mark.parametrize("victim", VICTIM_FILES)
def test_victim_file_leaves_src_identity_untouched(tmp_path: Path, victim: str) -> None:
    """Assert one #9387 victim file leaves src.* sys.modules untouched."""
    report = _run_victim(tmp_path, victim)

    missing = _missing_dependency(report)
    if missing is not None:
        pytest.skip(f"victim file needs unavailable dependency: {missing}")

    failures = [
        (classname, name)
        for classname, name, outcome in report["outcomes"]
        if outcome in ("failure", "error")
    ]
    allowed = _WORKTREE_ENV_FAILURES.get(victim, set())
    if (REPO_ROOT / ".git").is_file():
        # Worktree checkout: tolerate the two documented environment-
        # dependent validate_suite ids, fail on anything else.
        failures = [failure for failure in failures if failure not in allowed]
    elif any(failure in allowed for failure in failures):
        pytest.fail(
            f"{victim} failed ids that are only tolerated in a worktree "
            f"checkout: {failures}"
        )
    if failures:
        pytest.fail(
            f"{victim} had test failures inside the sentinel run "
            f"(exit={report['exit']}): {failures}"
        )

    _assert_namespace_unchanged(report, victim)
