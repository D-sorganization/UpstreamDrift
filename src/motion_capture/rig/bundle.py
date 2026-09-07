"""Session bundles: a recording plus everything needed to trust it later.

A bundle directory holds the plan that was recorded, one compressed video per
view, a ``recordings.json`` index describing each file, and the session
manifest with its outcome. Later stages (ingest, alignment, export) read the
bundle rather than the live cameras, so the bundle is the unit that travels,
is validated, and is archived.

Every successful recording is decode-probed when the index is built: frames,
duration and achieved rate are recorded, and a recording that covers less than
``MIN_COVERAGE`` of the requested duration or rate is ``degraded``. The rig's
first bring-up recordings were 20 % short while every exit code was zero;
that is the failure this exists to name.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .plan import CaptureMode, RigPlan
from .probe import Prober, RecordingProbe, probe_recording
from .recorder import RecordingResult
from .session import CameraStats, CaptureOutcome, SessionManifest, classify

logger = get_logger(__name__)

BUNDLE_SCHEMA_VERSION = "capture-session-bundle/1.1.0"
PLAN_FILE = "plan.json"
RECORDINGS_FILE = "recordings.json"
MANIFEST_FILE = "session_manifest.json"
MIN_COVERAGE = 0.9


class RecordingEntry(BaseModel):
    """One view's recording as written to ``recordings.json``."""

    model_config = ConfigDict(frozen=True)

    view: str
    identity: str
    file: str  # relative to the bundle directory
    bytes: int
    returncode: int | None
    requested_mode: CaptureMode
    requested_duration_s: float
    frames: int | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.bytes > 0

    @property
    def achieved_fps(self) -> float | None:
        if not self.frames or not self.duration_s:
            return None
        return self.frames / self.duration_s

    def coverage_reason(self) -> str | None:
        """Why this recording is short, or ``None`` when it meets the floors."""
        if self.duration_s is None or self.frames is None:
            return None  # not probed: cannot judge, do not invent
        floor_s = MIN_COVERAGE * self.requested_duration_s
        if self.duration_s < floor_s:
            return f"{self.duration_s:.2f} s recorded < {floor_s:.2f} s"
        fps = self.achieved_fps
        floor_fps = MIN_COVERAGE * self.requested_mode.fps
        if fps is not None and fps < floor_fps:
            return f"{fps:.1f} fps < {floor_fps:.1f} fps"
        return None


class RecordingsIndex(BaseModel):
    """``recordings.json``."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = BUNDLE_SCHEMA_VERSION
    duration_s: float
    recordings: tuple[RecordingEntry, ...]


class BundleCheck(BaseModel):
    """Result of :func:`check_bundle`."""

    model_config = ConfigDict(frozen=True)

    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def recording_stats(entry: RecordingEntry) -> CameraStats:
    """Fold a recording into the session's per-camera vocabulary.

    A recording is ``ok`` when ffmpeg exited 0 and wrote bytes; a zero-byte or
    failed recording is ``no_stream`` so the session outcome becomes ``blocked``
    rather than a quietly shorter dataset.
    """
    if not entry.ok:
        reason: str | None = (
            f"recorder exited {entry.returncode}"
            if entry.returncode not in (0, None)
            else "recording is empty"
        )
        state = "no_stream"
    elif (short := entry.coverage_reason()) is not None:
        state, reason = "degraded", short
    else:
        state, reason = "ok", None
    return CameraStats(
        view=entry.view,
        identity=entry.identity,
        requested_mode=entry.requested_mode,
        effective_mode=entry.requested_mode if entry.ok else None,
        frames=entry.frames or 0,
        achieved_fps=entry.achieved_fps or 0.0,
        state=state,
        reason=reason,
    )


def _probe_if_ok(result: RecordingResult, prober: Prober) -> RecordingProbe | None:
    """Probe only files a recorder actually wrote; failures carry no frame data."""
    return prober(result.path) if result.ok else None


def build_index(
    plan: RigPlan,
    results: list[RecordingResult],
    duration_s: float,
    bundle_dir: Path,
    *,
    prober: Prober = probe_recording,
) -> RecordingsIndex:
    """Describe ``results`` relative to ``bundle_dir``; one entry per plan view.

    Precondition: ``results`` carry exactly the plan's identities. Successful
    recordings are decode-probed through ``prober`` (injectable for tests).
    """
    require(duration_s > 0, "duration_s must be positive", duration_s)
    by_identity = {r.identity: r for r in results}
    require(
        set(by_identity) == {c.identity for c in plan.cameras},
        "results must cover exactly the plan cameras",
        sorted(set(by_identity) ^ {c.identity for c in plan.cameras}),
    )
    entries = []
    for binding in plan.cameras:
        result = by_identity[binding.identity]
        probe = _probe_if_ok(result, prober)
        entries.append(
            RecordingEntry(
                view=binding.view,
                identity=binding.identity,
                file=result.path.name
                if result.path.parent == bundle_dir
                else str(result.path),
                bytes=result.bytes_written,
                returncode=result.returncode,
                requested_mode=binding.mode,
                requested_duration_s=duration_s,
                frames=probe.frames if probe else None,
                duration_s=probe.duration_s if probe else None,
                width=probe.width if probe else None,
                height=probe.height if probe else None,
            )
        )
    return RecordingsIndex(duration_s=duration_s, recordings=tuple(entries))


def write_bundle(
    bundle_dir: Path,
    plan: RigPlan,
    index: RecordingsIndex,
    *,
    started_utc: str,
    tools_schema: Mapping[str, object] | None = None,
) -> SessionManifest:
    """Write ``plan.json``, ``recordings.json`` and the manifest; return the manifest.

    Postcondition: the three files exist and the manifest outcome follows
    :func:`session.classify` over the recordings.
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)
    plan.save(bundle_dir / PLAN_FILE)
    (bundle_dir / RECORDINGS_FILE).write_text(
        index.model_dump_json(indent=2), encoding="utf-8"
    )
    stats = [recording_stats(e) for e in index.recordings]
    outcome, reasons = classify(stats)
    manifest = SessionManifest(
        plan_name=plan.name,
        started_utc=started_utc,
        duration_s=index.duration_s,
        cameras=tuple(stats),
        outcome=outcome,
        reasons=reasons,
        tools_schema=dict(tools_schema or {}),
    )
    manifest.save(bundle_dir / MANIFEST_FILE)
    logger.info("session bundle %s: %s", bundle_dir, outcome.value)
    return manifest


def _load_json(
    path: Path, model: type[BaseModel], problems: list[str]
) -> BaseModel | None:
    if not path.is_file():
        problems.append(f"missing {path.name}")
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        problems.append(f"{path.name}: {exc.__class__.__name__}: {exc}")
        return None


def check_bundle(bundle_dir: Path) -> BundleCheck:
    """Validate a bundle on disk without opening any video.

    Checks: the three JSON files parse against their schemas, every plan view
    has a recording entry, every ``ok`` entry's file exists with the recorded
    size, and the manifest outcome is consistent with the recordings.
    """
    require(bundle_dir.is_dir(), "bundle_dir must be a directory", str(bundle_dir))
    problems: list[str] = []
    plan = _load_json(bundle_dir / PLAN_FILE, RigPlan, problems)
    index = _load_json(bundle_dir / RECORDINGS_FILE, RecordingsIndex, problems)
    manifest = _load_json(bundle_dir / MANIFEST_FILE, SessionManifest, problems)
    if isinstance(plan, RigPlan) and isinstance(index, RecordingsIndex):
        views = {c.view for c in plan.cameras}
        recorded = {e.view for e in index.recordings}
        for view in sorted(views - recorded):
            problems.append(f"view {view!r} has no recording entry")
        for entry in index.recordings:
            path = bundle_dir / entry.file
            if entry.ok and not path.is_file():
                problems.append(f"{entry.view}: file {entry.file} is missing")
            elif entry.ok and path.stat().st_size != entry.bytes:
                problems.append(f"{entry.view}: {entry.file} size differs from index")
    if isinstance(index, RecordingsIndex) and isinstance(manifest, SessionManifest):
        expected, _ = classify([recording_stats(e) for e in index.recordings])
        if manifest.outcome is not expected:
            problems.append(
                f"manifest outcome {manifest.outcome.value} != {expected.value} "
                "implied by recordings"
            )
        if manifest.outcome is CaptureOutcome.UNAVAILABLE and not manifest.reasons:
            problems.append("unavailable outcome without reasons")
    return BundleCheck(problems=tuple(problems))
