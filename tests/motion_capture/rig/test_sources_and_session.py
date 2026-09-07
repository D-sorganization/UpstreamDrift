"""Frame-source lifecycle and session outcomes, entirely without hardware."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.motion_capture.rig.plan import (
    CameraBinding,
    CameraControls,
    CaptureMode,
    RigPlan,
)
from src.motion_capture.rig.session import (
    MANIFEST_SCHEMA_VERSION,
    CameraStats,
    CaptureOutcome,
    CaptureSession,
    CaptureTuning,
    SessionManifest,
    classify,
)
from src.motion_capture.rig.sources import (
    HOST_MONOTONIC,
    FrameSource,
    SyntheticFrameSource,
)
from src.shared.python.core.contracts import StateError

pytestmark = pytest.mark.unit


def _plan(*views: str, fps: int = 60) -> RigPlan:
    return RigPlan(
        name="test",
        cameras=tuple(
            CameraBinding(view=v, serial=f"serial-{v}", mode=CaptureMode(fps=fps))
            for v in views
        ),
    )


# --------------------------------------------------------------------------- sources


def test_synthetic_source_is_a_frame_source_and_stamps_host_clock() -> None:
    src = SyntheticFrameSource("cam")
    assert isinstance(src, FrameSource)
    mode = src.open(CaptureMode(fps=100), CameraControls(exposure=-5))
    assert mode.fps == 100
    assert src.controls_applied == {"exposure": -5.0}
    first, second = src.read(), src.read()
    assert first is not None and second is not None
    assert (first.seq, second.seq) == (0, 1)
    assert second.t_ns - first.t_ns == 10_000_000  # 100 fps -> 10 ms
    assert first.clock_domain == HOST_MONOTONIC
    assert first.shape == (12, 16, 3)


def test_synthetic_source_read_before_open_is_a_state_error() -> None:
    with pytest.raises(StateError):
        SyntheticFrameSource("cam").read()


def test_synthetic_source_fault_injection_and_flash() -> None:
    src = SyntheticFrameSource("cam", fail_after=2, flash_at=1)
    src.open(CaptureMode())
    frames = [src.read() for _ in range(4)]
    assert frames[0] is not None and frames[0].image.max() == 32
    assert frames[1] is not None and frames[1].image.max() == 255
    assert frames[2] is None and frames[3] is None


# --------------------------------------------------------------------------- classify


def _stats(view: str, state: str, reason: str | None = None) -> CameraStats:
    return CameraStats(
        view=view,
        identity=view,
        requested_mode=CaptureMode(),
        state=state,
        reason=reason,
    )


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["ok", "ok", "ok"], CaptureOutcome.SUPPORTED),
        (["ok", "degraded", "ok"], CaptureOutcome.DEGRADED),
        (["ok", "no_stream", "ok"], CaptureOutcome.BLOCKED),
        (["ok", "open_failed", "degraded"], CaptureOutcome.BLOCKED),
        (["no_stream", "open_failed"], CaptureOutcome.UNAVAILABLE),
        ([], CaptureOutcome.UNAVAILABLE),
    ],
)
def test_classify_folds_camera_states_into_acceptance_outcomes(
    states: list[str], expected: CaptureOutcome
) -> None:
    outcome, reasons = classify(
        [_stats(f"v{i}", s, s if s != "ok" else None) for i, s in enumerate(states)]
    )
    assert outcome is expected
    assert (len(reasons) > 0) == (expected is not CaptureOutcome.SUPPORTED)


# --------------------------------------------------------------------------- session


def test_session_requires_sources_for_exactly_the_plan_views() -> None:
    plan = _plan("a", "b")
    with pytest.raises(Exception, match="exactly the plan views"):
        CaptureSession(plan, {"a": SyntheticFrameSource("a")}, duration_s=1)
    with pytest.raises(Exception, match="duration_s"):
        CaptureSession(
            plan,
            {"a": SyntheticFrameSource("a"), "b": SyntheticFrameSource("b")},
            duration_s=0,
        )


def test_session_supported_when_every_camera_meets_rate(tmp_path: Path) -> None:
    plan = _plan("a", "b", "c")
    sources = {v: SyntheticFrameSource(f"serial-{v}") for v in ("a", "b", "c")}
    manifest = CaptureSession(
        plan, sources, duration_s=5, tuning=CaptureTuning(max_frames=120)
    ).run()
    assert manifest.outcome is CaptureOutcome.SUPPORTED
    assert manifest.reasons == ()
    assert [c.view for c in manifest.cameras] == ["a", "b", "c"]
    for cam in manifest.cameras:
        assert cam.state == "ok"
        assert cam.frames == 120
        assert cam.achieved_fps == pytest.approx(60.0, rel=1e-6)
        assert cam.worst_gap_ms == pytest.approx(1000 / 60, rel=1e-6)
        assert cam.clock_domain == HOST_MONOTONIC
    path = manifest.save(tmp_path / "out" / "session_manifest.json")
    reloaded = SessionManifest.model_validate_json(path.read_text(encoding="utf-8"))
    assert reloaded == manifest
    assert reloaded.schema_version == MANIFEST_SCHEMA_VERSION


def test_session_degraded_when_a_camera_runs_slow() -> None:
    plan = _plan("a", "b")
    sources = {
        "a": SyntheticFrameSource("serial-a"),
        "b": SyntheticFrameSource("serial-b", fps=40),  # 40 < 0.9 * 60
    }
    manifest = CaptureSession(
        plan, sources, duration_s=5, tuning=CaptureTuning(max_frames=60)
    ).run()
    assert manifest.outcome is CaptureOutcome.DEGRADED
    slow = next(c for c in manifest.cameras if c.view == "b")
    assert slow.state == "degraded"
    assert slow.reason is not None and "40.0 fps" in slow.reason


def test_session_blocked_when_a_camera_loses_its_reservation() -> None:
    plan = _plan("a", "b")
    sources = {
        "a": SyntheticFrameSource("serial-a"),
        "b": SyntheticFrameSource("serial-b", fail_after=0),  # never delivers
    }
    session = CaptureSession(
        plan,
        sources,
        duration_s=2,
        tuning=CaptureTuning(max_frames=30, stall_reads=5, max_reopens=1),
    )
    manifest = session.run()
    assert manifest.outcome is CaptureOutcome.BLOCKED
    dead = next(c for c in manifest.cameras if c.view == "b")
    assert dead.state == "no_stream"
    assert dead.reopens == 1
    assert dead.failed_reads >= 5
    assert any("b (serial-b)" in r for r in manifest.reasons)


def test_session_unavailable_when_no_camera_opens() -> None:
    class Refusing:
        identity = "x"

        def open(
            self, mode: CaptureMode, controls: CameraControls | None = None
        ) -> CaptureMode:
            raise StateError("isochronous reservation refused")

        def read(self):  # pragma: no cover - never reached
            return None

        def close(self) -> None:
            pass

    plan = _plan("a")
    manifest = CaptureSession(plan, {"a": Refusing()}, duration_s=1).run()
    assert manifest.outcome is CaptureOutcome.UNAVAILABLE
    assert manifest.cameras[0].state == "open_failed"
    assert "reservation refused" in (manifest.cameras[0].reason or "")
