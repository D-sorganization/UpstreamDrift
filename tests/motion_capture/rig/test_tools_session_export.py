"""The Tools session export fails closed here and is exercised where Tools resolves.

The root test process deliberately keeps ``shared.python.sidekick.lab.mocap``
unresolvable (UpstreamDrift's own Sidekick is cached first), so in-process the
bridge must report that as data and refuse to export. The ready path — the
rig session read back through the Tools parser — runs in a fresh Tools-first
process via ``tests/fixtures/mocap_session_export/run_checks.py``, exactly as
the capture-rig calibration worker is checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.motion_capture.rig.plan import CameraBinding, RigPlan
from src.motion_capture.rig.session import (
    CaptureSession,
    CaptureTuning,
    SessionManifest,
)
from src.motion_capture.rig.sources import SyntheticFrameSource
from src.motion_capture.rig.tools_bridge import (
    MOCAP_SESSION_FILE,
    RecordingTerms,
    SchemaProbe,
    export_session_manifest,
    export_to_bundle,
    map_camera_records,
    probe_tools_schema,
)
from src.shared.python.core.contracts import StateError

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
RUN_CHECKS = ROOT / "tests/fixtures/mocap_session_export/run_checks.py"


def _plan() -> RigPlan:
    return RigPlan(name="bench", cameras=(CameraBinding(view="face", serial="SN-1"),))


def _captured(plan: RigPlan) -> SessionManifest:
    sources = {
        c.view: SyntheticFrameSource(c.identity, realtime=False) for c in plan.cameras
    }
    tuning = CaptureTuning(max_frames=3)
    return CaptureSession(plan, sources, duration_s=0.2, tuning=tuning).run()


def test_recording_terms_derive_no_store_honestly() -> None:
    assert RecordingTerms().no_store is True
    assert RecordingTerms(raw_video_retained=True).no_store is False
    assert RecordingTerms(retention_days=1).no_store is False


def test_export_fails_closed_when_the_pinned_schema_is_not_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan = _plan()
    manifest = _captured(plan)
    absent = SchemaProbe("unavailable", "not pinned")
    monkeypatch.setattr(
        "src.motion_capture.rig.tools_bridge.probe_tools_schema", lambda: absent
    )
    with pytest.raises(StateError, match="unavailable: not pinned"):
        export_session_manifest(manifest, plan)
    report = export_to_bundle(tmp_path, manifest, plan)
    assert report["status"] == "unavailable" and "not pinned" in str(report["reason"])
    assert report["path"] is None and not (tmp_path / MOCAP_SESSION_FILE).exists()


def test_root_test_process_keeps_the_tools_family_unresolved() -> None:
    """Pin the design the fixture below depends on; see src/__init__.py."""
    probe = probe_tools_schema()
    assert probe.status == "unavailable", probe
    assert export_to_bundle(Path("unused"), _captured(_plan()), _plan())["status"] == (
        "unavailable"
    )


@pytest.mark.integration
def test_ready_path_round_trips_through_tools_in_its_own_process() -> None:
    before = probe_tools_schema()
    result = subprocess.run(
        [sys.executable, "-I", str(RUN_CHECKS)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert probe_tools_schema() == before


def test_camera_records_check_their_inputs_before_touching_tools() -> None:
    plan = _plan()
    manifest = _captured(plan)
    with pytest.raises(ValueError, match="manifest must be a SessionManifest"):
        map_camera_records({"cameras": ()}, plan)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="plan must be a RigPlan"):
        map_camera_records(manifest, "bench")  # type: ignore[arg-type]


def test_camera_records_fail_closed_when_the_pinned_schema_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    absent = SchemaProbe("incompatible", "ships mocap-session/2.0.0")
    monkeypatch.setattr(
        "src.motion_capture.rig.tools_bridge.probe_tools_schema", lambda: absent
    )
    with pytest.raises(StateError, match="incompatible: ships mocap-session/2.0.0"):
        map_camera_records(_captured(plan), plan)
