"""A rig capture session projected onto the one Tools ``MocapSession`` contract.

ADR-0041 / #9422: UpstreamDrift consumes the pinned Tools schema; it does not
restate it. Every assertion here reads the export back through the *Tools*
parser, so the contract under test is the pinned one, not a local copy.

Runs under ``run_checks.py`` with the Tools family resolved first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shared.python.sidekick.lab.mocap import (
    MOCAP_SESSION_SCHEMA_VERSION,
    SessionState,
    load_session_manifest,
)

from src.motion_capture.rig import __main__ as cli
from src.motion_capture.rig.bundle import MANIFEST_FILE
from src.motion_capture.rig.plan import CameraBinding, RigPlan
from src.motion_capture.rig.session import (
    CameraStats,
    CaptureOutcome,
    CaptureSession,
    CaptureTuning,
    SessionManifest,
)
from src.motion_capture.rig.sources import SyntheticFrameSource
from src.motion_capture.rig.tools_bridge import (
    MOCAP_SESSION_FILE,
    PINNED_SESSION_SCHEMA_VERSION,
    RecordingTerms,
    export_session_manifest,
    export_to_bundle,
    probe_tools_schema,
)


def _plan() -> RigPlan:
    return RigPlan(
        name="bench",
        cameras=(
            CameraBinding(view="face", serial="SN-1"),
            CameraBinding(view="down", port_path="1-2.3"),
        ),
    )


def _captured(plan: RigPlan) -> SessionManifest:
    sources = {
        c.view: SyntheticFrameSource(c.identity, realtime=False) for c in plan.cameras
    }
    tuning = CaptureTuning(max_frames=5)
    return CaptureSession(plan, sources, duration_s=0.2, tuning=tuning).run()


def test_probe_reports_the_pinned_schema_as_ready() -> None:
    probe = probe_tools_schema()
    assert probe.status == "ready", probe.reason
    assert (
        probe.version == MOCAP_SESSION_SCHEMA_VERSION == PINNED_SESSION_SCHEMA_VERSION
    )


def test_export_round_trips_through_the_tools_parser() -> None:
    plan = _plan()
    manifest = _captured(plan)
    text = export_session_manifest(manifest, plan)

    session = load_session_manifest(text)  # the Tools contract, not a UD copy
    assert session.schema_version == MOCAP_SESSION_SCHEMA_VERSION
    assert session.session_id == f"bench@{manifest.started_utc}"
    assert session.created_at_utc == manifest.started_utc
    assert session.world_frame.frame_id == "affinedrift-world-v1"
    assert [c.device_id for c in session.cameras] == ["SN-1", "1-2.3"]
    assert [c.serial_number for c in session.cameras] == ["SN-1", None]
    assert {c.provider_id for c in session.cameras} == {
        "upstreamdrift.motion_capture.rig"
    }
    (clock,) = session.clocks
    assert clock.clock_id == "host_monotonic_ns" and clock.kind == "host-monotonic"
    (method,) = session.methods
    assert method.method_id == "upstreamdrift.motion_capture.rig.capture"
    assert method.license_spdx == "MIT"
    # No consent or calibration was recorded, so the session is not finalized.
    assert session.state is SessionState.INCOMPLETE
    assert session.recording_policy.no_store is True
    assert "no consent recorded" in session.warnings
    assert "no calibration recorded" in session.warnings
    # Canonical form: sorted keys, compact, one trailing newline, deterministic.
    assert text.endswith("\n") and text == export_session_manifest(manifest, plan)
    assert list(json.loads(text)) == sorted(json.loads(text))


def test_export_finalizes_only_a_supported_consented_calibrated_session() -> None:
    plan = _plan()
    manifest = _captured(plan)
    assert manifest.outcome is CaptureOutcome.SUPPORTED
    terms = RecordingTerms(consent_recorded=True, raw_video_retained=True)
    text = export_session_manifest(
        manifest, plan, terms, calibration_ids=("intrinsics-2026-09-19",)
    )
    session = load_session_manifest(text)
    assert session.state is SessionState.FINALIZED
    assert session.calibration_ids == ("intrinsics-2026-09-19",)
    assert session.recording_policy.raw_video_retained is True
    assert session.recording_policy.no_store is False
    assert session.warnings == ()


def test_export_refuses_retained_video_without_consent() -> None:
    plan = _plan()
    terms = RecordingTerms(consent_recorded=False, raw_video_retained=True)
    with pytest.raises(ValueError, match="consent"):
        export_session_manifest(_captured(plan), plan, terms)


def test_export_refuses_a_camera_the_plan_does_not_bind() -> None:
    plan = _plan()
    manifest = _captured(plan)
    stray = CameraStats(view="ghost", identity="X", requested_mode=plan.cameras[0].mode)
    stray_manifest = manifest.model_copy(update={"cameras": (*manifest.cameras, stray)})
    with pytest.raises(ValueError, match="ghost"):
        export_session_manifest(stray_manifest, plan)


def test_export_to_bundle_reports_rejections_as_data(tmp_path: Path) -> None:
    plan = _plan()
    terms = RecordingTerms(consent_recorded=False, raw_video_retained=True)
    report = export_to_bundle(tmp_path, _captured(plan), plan, terms)
    assert report["status"] == "rejected" and "consent" in str(report["reason"])
    assert not (tmp_path / MOCAP_SESSION_FILE).exists()


def test_cli_capture_writes_the_canonical_session_beside_the_rig_manifest(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _plan().save(plan_path)
    out = tmp_path / "out"
    args = ["capture", "--plan", str(plan_path), "--duration", "0.3", "--out", str(out)]
    assert cli.main([*args, "--synthetic"]) == 0
    rig_manifest = SessionManifest.model_validate_json(
        (out / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert rig_manifest.tools_schema["status"] == "ready"
    assert rig_manifest.tools_schema["export"]["status"] == "written"
    session = load_session_manifest((out / MOCAP_SESSION_FILE).read_text("utf-8"))
    assert session.state is SessionState.INCOMPLETE
    assert session.recording_policy.no_store is True


def test_cli_record_dry_run_needs_no_consent_but_a_real_take_does(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    _plan().save(plan_path)
    out = tmp_path / "session"
    args = ["record", "--plan", str(plan_path), "--duration", "1", "--out", str(out)]
    assert cli.main([*args, "--dry-run"]) == 2  # NullRecorder: nothing captured
    rig_manifest = SessionManifest.model_validate_json(
        (out / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    assert rig_manifest.tools_schema["export"]["status"] == "written"
    session = load_session_manifest((out / MOCAP_SESSION_FILE).read_text("utf-8"))
    assert session.state is SessionState.INCOMPLETE
    assert session.recording_policy.raw_video_retained is False
    assert session.warnings[0].startswith("face")  # the classify reasons travel
    assert cli.main(["session-check", "--session", str(out)]) == 0
