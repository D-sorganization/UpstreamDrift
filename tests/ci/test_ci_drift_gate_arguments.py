"""Behavioral contracts for CI drift-gate test selection."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
pytestmark = pytest.mark.unit


def _drift_gate_run_block() -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == "Run Launch-Monitor Drift Gates (ADR-0046 G0)":
                return step["run"]
    raise AssertionError("Launch-Monitor Drift Gates step is missing")


@pytest.mark.parametrize("include_putting_green", [False, True])
@pytest.mark.skipif(os.name == "nt", reason="the extracted workflow step requires Bash")
def test_drift_gate_passes_each_selected_suite_as_a_separate_argument(
    tmp_path: Path, include_putting_green: bool
) -> None:
    """The optional suite is selected by directory presence without word splitting."""
    optional_suite = tmp_path / "tests/integration/putting_green_drift"
    if include_putting_green:
        optional_suite.mkdir(parents=True)

    capture = tmp_path / "argv.json"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"
    environment["CAPTURE"] = str(capture)
    subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _drift_gate_run_block()],
        cwd=tmp_path,
        env=environment,
        check=True,
        timeout=15,
    )

    arguments = json.loads(capture.read_text(encoding="utf-8"))
    expected_suites = ["tests/integration/launch_monitor_drift/"]
    if include_putting_green:
        expected_suites.append("tests/integration/putting_green_drift/")
    assert arguments == [
        "-m",
        "pytest",
        *expected_suites,
        "--timeout=120",
        "--timeout-method=thread",
        "-v",
        "--tb=short",
    ]
