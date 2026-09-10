"""Actual isolated provider execution leaves the capture application's imports intact."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from src.motion_capture.rig.tools_bridge import probe_tools_schema

ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "src/tools/capture_rig/reference_calibration/worker.py"
pytestmark = pytest.mark.integration


def test_reference_session_checks_in_actual_worker_environment() -> None:
    before = probe_tools_schema()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "tests/fixtures/reference_calibration/run_checks.py"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert probe_tools_schema() == before


def test_catalog_request_uses_tools_without_loading_it_into_parent(
    tmp_path: Path,
) -> None:
    before = probe_tools_schema()
    result = subprocess.run(
        [sys.executable, "-I", str(WORKER)],
        cwd=tmp_path,
        input=json.dumps(
            {"schema_version": "capture-reference-worker/1", "action": "catalog"}
        ),
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    targets = {
        target["reference_id"]: target for target in response["result"]["targets"]
    }
    assert targets["us-letter"]["object_points_m"][2] == [0.2794, 0.0, 0.2159]
    assert targets["yardstick"]["object_points_m"][-1][0] == 0.9144
    assert probe_tools_schema() == before


def test_unknown_worker_request_is_actionable() -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(WORKER)],
        input='{"schema_version":"unexpected","action":"catalog"}',
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    response = json.loads(result.stdout)
    assert response["ok"] is False
    assert "request schema" in response["error"]
