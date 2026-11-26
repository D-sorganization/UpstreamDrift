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


def _workflow(path: Path) -> dict[str, Any]:
    """Return a parsed workflow document."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must be a workflow mapping")
    return loaded


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """Return the workflow ``on`` mapping.

    PyYAML parses the bare ``on`` key as ``True``.
    """
    triggers = workflow.get(True, workflow.get("on"))
    if not isinstance(triggers, dict):
        raise TypeError("workflow triggers must be a mapping")
    return triggers


def _step(workflow: dict[str, Any], job: str, name: str) -> dict[str, Any]:
    """Return one named step from a job."""
    steps = workflow["jobs"][job]["steps"]
    found = next((step for step in steps if step.get("name") == name), None)
    if not isinstance(found, dict):
        raise AssertionError(f"missing step {name!r} in {job}")
    return found


def test_push_paths_ignore_matches_docs_only_patterns() -> None:
    """A ``.github`` push to main must still run CI Standard."""
    classifier = _load_module("classify_event_tier_15765", CLASSIFIER)
    ignored = _triggers(_workflow(CI_STANDARD))["push"]["paths-ignore"]
    assert ignored == list(classifier.DOCS_ONLY_PATTERNS)
    assert ".github/**" not in ignored


def test_code_quality_check_propagates_its_exit_status() -> None:
    """The code-quality step must not turn checker failures into warnings."""
    step = _step(_workflow(CI_STANDARD), "code-quality", "Code Quality Check")
    script = str(step.get("run", ""))
    assert "set -euo pipefail" in script
    assert "run_code_quality_check.py" in script
    assert "||" not in script


def test_osv_export_distinguishes_findings_from_crashes() -> None:
    """OSV exit 1 stays reportable and any higher status fails the monitor."""
    step = _step(
        _workflow(OSV_MONITOR), "osv-monitor", "Export OSV JSON for SLA triage"
    )
    script = str(step.get("run", ""))
    assert "|| true" not in script
    assert "||true" not in script.replace(" ", "")
    assert "--classify-exit" in script
    assert 'scan_class" = "crash"' in script or 'scan_class" == "crash"' in script
    assert 'triage_class" = "crash"' in script or 'triage_class" == "crash"' in script
    assert 'exit "$scan_status"' in script
    assert 'exit "$triage_status"' in script


def test_classify_osv_exit_separates_findings_from_crashes() -> None:
    """Exit 0 is clean, exit 1 is a finding, and every other status is a crash."""
    triage = _load_module("triage_osv_findings_15765", TRIAGE)
    assert triage.classify_osv_exit(0) == "clean"
    assert triage.classify_osv_exit(1) == "vulnerabilities"
    assert triage.classify_osv_exit(2) == "crash"
    assert triage.classify_osv_exit(127) == "crash"
    with pytest.raises(TypeError, match="status must be an int"):
        triage.classify_osv_exit(True)


def test_triage_crash_exits_above_the_finding_status(tmp_path: Path) -> None:
    """A missing OSV report is a tool crash, not a vulnerability finding."""
    missing = tmp_path / "missing.json"
    result = subprocess.run(
        [sys.executable, str(TRIAGE), "--osv-json", str(missing)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode > 1


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
    result = subprocess.run(
        [sys.executable, str(TRIAGE), "--osv-json", str(report)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "GHSA-test" in result.stdout


def test_code_quality_runner_returns_the_checker_status(tmp_path: Path) -> None:
    """A checker failure must be the runner's own exit status."""
    sample = tmp_path / "sample.py"
    sample.write_text("def helper():\n    return 1\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(QUALITY_RUNNER), "--changed-file", str(sample)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr


def test_code_quality_runner_accepts_a_clean_file(tmp_path: Path) -> None:
    """A documented function does not fail the code-quality runner."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        'def helper() -> int:\n    """Return one."""\n    return 1\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(QUALITY_RUNNER), "--changed-file", str(sample)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
