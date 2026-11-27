"""CI guards must fail closed on workflow, quality, and OSV crashes (#15765)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github/workflows/ci-standard.yml"
OSV_MONITOR = REPO_ROOT / ".github/workflows/security-osv-monitor.yml"
CLASSIFIER = REPO_ROOT / "scripts/ci/classify_event_tier.py"
TRIAGE = REPO_ROOT / "scripts/security/triage_osv_findings.py"
QUALITY_RUNNER = REPO_ROOT / "scripts/ci/run_code_quality_check.py"


def _load_module(name: str, path: Path) -> Any:
    """Load a repository script by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _step(workflow_path: Path, job: str, name: str) -> dict[str, Any]:
    """Return one named step from a workflow job."""
    loaded = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    found = next(
        step for step in loaded["jobs"][job]["steps"] if step.get("name") == name
    )
    if not isinstance(found, dict):
        raise AssertionError(f"missing step {name!r}")
    return found


def test_push_paths_ignore_matches_docs_only_patterns() -> None:
    """A ``.github`` push to main must still run CI Standard."""
    classifier = _load_module("classify_event_tier_15765", CLASSIFIER)
    loaded = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    triggers = loaded.get(True, loaded.get("on"))
    ignored = triggers["push"]["paths-ignore"]
    assert ignored == list(classifier.DOCS_ONLY_PATTERNS)
    assert ".github/**" not in ignored


def test_code_quality_check_propagates_its_exit_status() -> None:
    """The code-quality step must not turn checker failures into warnings."""
    script = str(_step(CI_STANDARD, "code-quality", "Code Quality Check")["run"])
    assert "set -euo pipefail" in script
    assert "run_code_quality_check.py" in script
    assert "||" not in script


def test_osv_export_distinguishes_findings_from_crashes() -> None:
    """OSV exit 1 stays reportable and any higher status fails the monitor."""
    script = str(
        _step(OSV_MONITOR, "osv-monitor", "Export OSV JSON for SLA triage")["run"]
    )
    assert "|| true" not in script
    assert "--classify-exit" in script
    assert '[ "$scan_class" = "crash" ]' in script
    assert '[ "$triage_class" = "crash" ]' in script
    assert 'exit "$scan_status"' in script
    assert 'exit "$triage_status"' in script


def test_classify_osv_exit_separates_findings_from_crashes() -> None:
    """Exit 0 is clean, exit 1 is a finding, and every other status is a crash."""
    triage = _load_module("triage_osv_findings_15765", TRIAGE)
    assert [triage.classify_osv_exit(status) for status in (0, 1, 2, 127)] == [
        "clean",
        "vulnerabilities",
        "crash",
        "crash",
    ]
    with pytest.raises(TypeError, match="status must be an int"):
        triage.classify_osv_exit(True)


def _run_triage(argument: str, value: str) -> subprocess.CompletedProcess[str]:
    """Run the OSV triage CLI and return the completed process."""
    return subprocess.run(
        [sys.executable, str(TRIAGE), argument, value],
        capture_output=True,
        text=True,
        check=False,
    )


def test_triage_crash_exits_above_the_finding_status(tmp_path: Path) -> None:
    """A missing OSV report is a tool crash, not a vulnerability finding."""
    assert _run_triage("--osv-json", str(tmp_path / "missing.json")).returncode > 1


def test_triage_findings_remain_exit_one(tmp_path: Path) -> None:
    """Summarized findings stay on exit 1 so the monitor can publish them."""
    report = tmp_path / "osv.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "packages": [
                            {
                                "package": {"name": "numpy"},
                                "vulnerabilities": [{"id": "GHSA-test"}],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result = _run_triage("--osv-json", str(report))
    assert result.returncode == 1
    assert "GHSA-test" in result.stdout


def _checker_stub(directory: Path, status: int) -> Path:
    """Write a checker stub that exits with ``status``."""
    stub = directory / f"checker_{status}.py"
    stub.write_text(f"import sys\nraise SystemExit({status})\n", encoding="utf-8")
    return stub


@pytest.mark.parametrize("status", [1, 0])
def test_code_quality_runner_preserves_checker_status(
    tmp_path: Path, status: int
) -> None:
    """The runner's exit status is the checker status, including zero."""
    sample = tmp_path / "sample.py"
    sample.write_text("x = 1\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(QUALITY_RUNNER),
            "--checker",
            str(_checker_stub(tmp_path, status)),
            "--changed-file",
            str(sample),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stderr


def test_review_events_diff_and_other_events_audit_the_tree() -> None:
    """Pull requests and merge groups diff; other events keep the full audit."""
    runner = _load_module("run_code_quality_check_15765", QUALITY_RUNNER)
    assert runner.CHECKER == REPO_ROOT / "src/tools/code_quality_check.py"
    cases = {
        ("pull_request", "main", "", ""): "origin/main",
        ("merge_group", "", "", "abc123"): "abc123",
        ("schedule", "", "", ""): None,
        ("push", "", "0" * 40, ""): None,
    }
    for (event, base_ref, before_sha, merge_base), expected in cases.items():
        assert (
            runner.diff_base_for_event(
                event,
                base_ref=base_ref,
                before_sha=before_sha,
                merge_group_base_sha=merge_base,
            )
            == expected
        )
    with pytest.raises(ValueError, match="pull_request requires base_ref"):
        runner.diff_base_for_event(
            "pull_request",
            base_ref="",
            before_sha="",
            merge_group_base_sha="",
        )
