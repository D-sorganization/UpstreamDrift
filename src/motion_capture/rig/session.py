"""Barrier-started multi-camera capture with typed outcomes.

The failure mode this guards against is silent and asymmetric: when two
cameras contend for one USB 2.0 root port, the winner looks perfect and the
loser produces nothing, and no error is raised anywhere. A session therefore
measures every camera against its requested rate and reports one of the
acceptance-program outcomes — ``supported``, ``degraded``, ``blocked``,
``unavailable`` — with a reason per camera, instead of a single boolean.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.core.contracts import StateError, require
from src.shared.python.logging_pkg.logging_config import get_logger

from .plan import CameraBinding, CaptureMode, RigPlan
from .sources import HOST_MONOTONIC, Frame, FrameSource

logger = get_logger(__name__)

MANIFEST_SCHEMA_VERSION = "capture-session-manifest/1.0.0"


class CaptureOutcome(str, Enum):
    """Acceptance-program vocabulary (docs/motion_capture/markerless_mocap_acceptance.md)."""

    SUPPORTED = "supported"
    DEGRADED = "degraded"
    BLOCKED = "blocked"
    UNAVAILABLE = "unavailable"


class CameraStats(BaseModel):
    """What one camera actually delivered."""

    model_config = ConfigDict(frozen=True)

    view: str
    identity: str
    requested_mode: CaptureMode
    effective_mode: CaptureMode | None = None
    frames: int = 0
    failed_reads: int = 0
    reopens: int = 0
    achieved_fps: float = 0.0
    worst_gap_ms: float = 0.0
    first_t_ns: int | None = None
    last_t_ns: int | None = None
    clock_domain: str = HOST_MONOTONIC
    state: str = "no_stream"  # ok | degraded | no_stream | open_failed
    reason: str | None = None


class SessionManifest(BaseModel):
    """Everything a later stage needs to trust — or distrust — this capture."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = MANIFEST_SCHEMA_VERSION
    plan_name: str
    started_utc: str
    duration_s: float
    cameras: tuple[CameraStats, ...]
    outcome: CaptureOutcome
    reasons: tuple[str, ...] = ()
    timing: dict[str, Any] = Field(default_factory=dict)  # filled by the sync stage
    tools_schema: dict[str, Any] = Field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


class _Worker:
    """Reads one source until the deadline, tracking stats and reopening on stall."""

    def __init__(
        self,
        binding: CameraBinding,
        source: FrameSource,
        *,
        stall_reads: int,
        max_reopens: int,
        max_frames: int | None,
    ) -> None:
        self.binding = binding
        self.source = source
        self.stall_reads = stall_reads
        self.max_reopens = max_reopens
        self.max_frames = max_frames
        self.effective: CaptureMode | None = None
        self.frames = self.failed = self.reopens = 0
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self.worst_gap_ns = 0
        self.error: str | None = None

    @property
    def nominal_fps(self) -> int:
        """Requested frame rate; delegates so callers need not reach through."""
        return self.binding.mode.fps

    def open(self) -> bool:
        try:
            self.effective = self.source.open(self.binding.mode, self.binding.controls)
        except (StateError, OSError, RuntimeError, ValueError) as exc:
            self.error = f"open failed: {exc}"
            logger.warning("camera %s: %s", self.binding.identity, self.error)
            return False
        return True

    def run(self, barrier: threading.Barrier, deadline_ns: int) -> None:
        barrier.wait()
        consecutive = 0
        while time.monotonic_ns() < deadline_ns:
            if self.max_frames is not None and self.frames >= self.max_frames:
                break
            frame = self.source.read()
            if frame is None:
                self.failed += 1
                consecutive += 1
                if consecutive >= self.stall_reads and not self._reopen():
                    break
                continue
            consecutive = 0
            self._record(frame)
        self.source.close()

    def _reopen(self) -> bool:
        if self.reopens >= self.max_reopens:
            self.error = f"stalled after {self.frames} frames; reopen budget exhausted"
            return False
        self.reopens += 1
        logger.warning("camera %s stalled; reopening", self.binding.identity)
        return self.open()

    def _record(self, frame: Frame) -> None:
        if self.first_ns is None:
            self.first_ns = frame.t_ns
        elif self.last_ns is not None:
            self.worst_gap_ns = max(self.worst_gap_ns, frame.t_ns - self.last_ns)
        self.last_ns = frame.t_ns
        self.frames += 1

    def stats(self, min_fps_ratio: float) -> CameraStats:
        fps = 0.0
        if self.first_ns is not None and self.last_ns is not None and self.frames > 1:
            fps = (self.frames - 1) / ((self.last_ns - self.first_ns) / 1e9)
        target = self.nominal_fps
        if self.effective is None:
            state, reason = "open_failed", self.error
        elif self.frames == 0:
            state, reason = "no_stream", self.error or "opened but delivered no frames"
        elif fps < min_fps_ratio * target:
            state, reason = (
                "degraded",
                f"{fps:.1f} fps < {min_fps_ratio:.0%} of {target}",
            )
        else:
            state, reason = "ok", None
        return CameraStats(
            view=self.binding.view,
            identity=self.binding.identity,
            requested_mode=self.binding.mode,
            effective_mode=self.effective,
            frames=self.frames,
            failed_reads=self.failed,
            reopens=self.reopens,
            achieved_fps=fps,
            worst_gap_ms=self.worst_gap_ns / 1e6,
            first_t_ns=self.first_ns,
            last_t_ns=self.last_ns,
            state=state,
            reason=reason,
        )


def classify(stats: list[CameraStats]) -> tuple[CaptureOutcome, tuple[str, ...]]:
    """Fold per-camera states into one outcome plus the reasons behind it."""
    states = [s.state for s in stats]
    reasons = tuple(f"{s.view} ({s.identity}): {s.reason}" for s in stats if s.reason)
    if not stats or all(st in ("no_stream", "open_failed") for st in states):
        return CaptureOutcome.UNAVAILABLE, reasons or ("no camera delivered frames",)
    if any(st in ("no_stream", "open_failed") for st in states):
        return CaptureOutcome.BLOCKED, reasons
    if any(st == "degraded" for st in states):
        return CaptureOutcome.DEGRADED, reasons
    return CaptureOutcome.SUPPORTED, ()


class CaptureSession:
    """Open every planned camera, start them together, measure, classify.

    Preconditions: ``sources`` maps exactly the plan's views; ``duration_s`` > 0;
    ``0 < min_fps_ratio <= 1``. Postcondition: the manifest lists one
    :class:`CameraStats` per plan view, in plan order.
    """

    def __init__(
        self,
        plan: RigPlan,
        sources: Mapping[str, FrameSource],
        *,
        duration_s: float,
        min_fps_ratio: float = 0.9,
        settle_s: float = 0.0,
        stall_reads: int = 30,
        max_reopens: int = 1,
        max_frames: int | None = None,
    ) -> None:
        views = {c.view for c in plan.cameras}
        require(
            set(sources) == views,
            "sources must cover exactly the plan views",
            sorted(views ^ set(sources)),
        )
        require(duration_s > 0, "duration_s must be positive", duration_s)
        require(
            0 < min_fps_ratio <= 1, "min_fps_ratio must be in (0, 1]", min_fps_ratio
        )
        require(settle_s >= 0, "settle_s must be non-negative", settle_s)
        self.plan = plan
        self.duration_s = duration_s
        self.min_fps_ratio = min_fps_ratio
        self.settle_s = settle_s
        self._workers = [
            _Worker(
                b,
                sources[b.view],
                stall_reads=stall_reads,
                max_reopens=max_reopens,
                max_frames=max_frames,
            )
            for b in plan.cameras
        ]

    def run(self) -> SessionManifest:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        live = []
        for worker in self._workers:
            if worker.open():
                live.append(worker)
            time.sleep(
                self.settle_s
            )  # MSMF teardown is asynchronous (bring-up finding 6)
        if live:
            self._run_live(live)
        stats = [w.stats(self.min_fps_ratio) for w in self._workers]
        outcome, reasons = classify(stats)
        logger.info("capture %s: %s", self.plan.name, outcome.value)
        return SessionManifest(
            plan_name=self.plan.name,
            started_utc=started,
            duration_s=self.duration_s,
            cameras=tuple(stats),
            outcome=outcome,
            reasons=reasons,
        )

    def _run_live(self, live: list[_Worker]) -> None:
        barrier = threading.Barrier(len(live))
        deadline = time.monotonic_ns() + int(self.duration_s * 1e9)
        threads = [
            threading.Thread(
                target=w.run, args=(barrier, deadline), name=w.binding.view
            )
            for w in live
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
