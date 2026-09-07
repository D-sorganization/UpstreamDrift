"""Session bundles from existing video files (#9659).

The chain after recording (proxy, ingest, compare, reconstruct) reads a
bundle, not cameras. This builds the same three documents around files the
operator already has — one file for a single-camera session, several for a
multi-camera one — so imported footage flows through unchanged code. Each
file is decode-probed for frames, duration, size and rate; the probe is the
capture mode recorded in the plan, and the binding identity is the file name
so the bundle says where every view came from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .bundle import SessionManifest, build_index, write_bundle
from .plan import CameraBinding, CaptureMode, RigPlan
from .probe import RecordingProbe, probe_recording
from .recorder import RecordingResult

logger = get_logger(__name__)

Prober = Callable[[Path], RecordingProbe]
IMPORT_PREFIX = "file:"


def parse_view_spec(text: str) -> tuple[str, Path]:
    """``NAME=PATH`` → ``(name, path)``; precondition: both parts non-empty."""
    name, sep, path = text.partition("=")
    require(
        sep == "=" and name.strip() and path.strip(), "view must be NAME=PATH", text
    )
    return name.strip(), Path(path.strip())


def _binding(view: str, path: Path, probe: RecordingProbe) -> CameraBinding:
    rate = probe.nominal_fps or (
        probe.frames / probe.duration_s if probe.duration_s else 0
    )
    fps = int(round(rate)) if rate and rate > 0 else 30
    mode = CaptureMode(
        width=max(int(probe.width or 0), 1),
        height=max(int(probe.height or 0), 1),
        fps=max(fps, 1),
        fourcc="FILE",
    )
    return CameraBinding(view=view, serial=f"{IMPORT_PREFIX}{path.name}", mode=mode)


def import_videos(
    views: Mapping[str, Path] | Sequence[tuple[str, Path]],
    out_dir: Path,
    *,
    prober: Prober = probe_recording,
    plan_name: str | None = None,
) -> SessionManifest:
    """Write a bundle in ``out_dir`` describing ``views``; return its manifest.

    Preconditions: at least one view; unique view names; every file exists.
    Postcondition: ``recordings.json`` carries each file's probe and the
    absolute path, so nothing is copied.
    """
    items = list(views.items()) if isinstance(views, Mapping) else list(views)
    require(bool(items), "import needs at least one view")
    names = [n for n, _ in items]
    require(len(set(names)) == len(names), "view names must be unique", names)
    for _, path in items:
        require(path.is_file(), "video file must exist", str(path))
    probes = {path.resolve(): prober(path) for _, path in items}
    bindings = tuple(_binding(n, p, probes[p.resolve()]) for n, p in items)
    plan = RigPlan(
        name=plan_name or f"import:{out_dir.name}",
        cameras=bindings,
        notes="imported from files; modes are the probed streams",
    )
    results = [
        RecordingResult(b.identity, p.resolve(), 0, p.stat().st_size)
        for b, (_, p) in zip(bindings, items, strict=True)
    ]
    duration = max(pr.duration_s for pr in probes.values()) or 1.0
    index = build_index(
        plan, results, duration, out_dir, prober=lambda p: probes[p.resolve()]
    )
    manifest = write_bundle(
        out_dir,
        plan,
        index,
        started_utc=datetime.now(UTC).isoformat(),
        tools_schema={"imported": True},
    )
    logger.info("imported %d view(s) into %s", len(items), out_dir)
    return manifest
