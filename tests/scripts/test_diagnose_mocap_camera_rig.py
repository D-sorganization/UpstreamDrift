"""The rig diagnostic is a thin CLI over ``motion_capture.rig``.

The topology, plan, and session logic it relies on is covered in
``tests/motion_capture/rig``; this file pins the script's own seams.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from src.motion_capture.rig.plan import CaptureMode
from src.motion_capture.rig.topology import CameraLocation

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "diagnose_mocap_camera_rig.py"


@pytest.fixture(scope="module")
def rig_script():
    spec = importlib.util.spec_from_file_location("diagnose_mocap_camera_rig", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def _cam(identity: str, serial: str | None, index: int | None) -> CameraLocation:
    return CameraLocation(
        camera=f"USB\\VID_32E4&PID_5234&MI_00\\{identity}",
        composite=None,
        serial=serial,
        identity=identity,
        root_hub=None,
        root_port=None,
        host=None,
        hub_depth=0,
        index=index,
    )


def test_help_runs_without_hardware() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--concurrent-seconds" in result.stdout


def test_plan_from_topology_binds_serial_or_port_path_and_skips_unindexed(
    rig_script,
) -> None:
    cams = [
        _cam("2605160001", "2605160001", 1),
        _cam("path_D-D35A8F7-0-0000", None, 2),
        _cam("ghost", None, None),
    ]
    plan = rig_script._plan_from_topology(cams, CaptureMode(fps=30))
    assert [c.view for c in plan.cameras] == ["2605160001", "path_D-D35A8F7-0-0000"]
    assert plan.cameras[0].serial == "2605160001" and plan.cameras[0].port_path is None
    assert plan.cameras[1].serial is None
    assert plan.cameras[1].port_path == "path_D-D35A8F7-0-0000"
    assert all(c.mode.fps == 30 for c in plan.cameras)
