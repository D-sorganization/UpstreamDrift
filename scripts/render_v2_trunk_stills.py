"""Address, top and impact stills of the full-body-v2 trunk visuals (HRG-D3, #13174).

Run through the showcase CLI (``--v2-spec SPEC.json --swing driver=BUNDLE.npz``).
Each still carries spec sha, event and frame; the manifest records the command.
A viewer that cannot render, or a blank frame, is an error, never a placeholder.
Stills are visual evidence, not physics validation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    from scripts.render_golfer_realism_showcase import ShowcasePlan
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import (
        ExportSettings,
        NativeBackend,
        SwingInput,
    )

EVENTS: Final[tuple[str, ...]] = ("address", "top", "impact")
MANIFEST_SCHEMA: Final[str] = "v2-trunk-stills-manifest/v1"
SHA_CHARS: Final[int] = 12  # spec sha256 prefix in filenames and labels


class StillsError(RuntimeError):
    """A still cannot be produced here (no viewer, or an unusable frame)."""


def load_v2_spec(path: Path) -> dict[str, Any]:
    """Read ``path`` as a full-body-v2 document (``ValueError`` if it is not)."""
    from src.shared.python.motion_matching import full_body_schema, full_body_v2

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    full_body_schema.require_full_body_schema(
        document,
        consumer="the v2 trunk stills",
        accepted=(full_body_v2.FULL_BODY_V2_SCHEMA_VERSION,),
    )
    full_body_v2.trunk_bodies(document)
    return dict(document)


def detect_event_frames(swing: SwingInput) -> dict[str, int]:
    """State index of address (first), top and impact of ``swing``.

    Top and impact use the shared ``club_face`` rules on the MuJoCo face-centre
    path (the exporter's impact detector); impact rounds to the nearer state.
    Raises ``BackendUnavailable`` without MuJoCo, ``ValueError`` if none is valid.
    """
    from src.shared.python.model_appearance.club_face import (
        ball_passage,
        top_of_backswing_index,
    )
    from src.tools.native_viewer_export.overlay import face_centre_trajectory

    times, head = face_centre_trajectory(swing)
    _, segment, fraction = ball_passage(times, head)
    return {
        "address": 0,
        "top": int(top_of_backswing_index(times, head)),
        "impact": int(segment) + int(fraction >= 0.5),
    }


def _checked_frames(found: Mapping[str, int], n_states: int) -> dict[str, int]:
    frames = {event: int(found[event]) for event in EVENTS}
    order = list(frames.values())
    if order[0] < 0 or order[-1] >= n_states or sorted(set(order)) != order:
        raise ValueError(
            f"event frames {frames} must increase address < top < impact inside "
            f"the {n_states} states of the swing"
        )
    return frames


def _ready_backends(
    plan: ShowcasePlan, factory: Callable[[str], NativeBackend]
) -> dict[str, NativeBackend]:
    backends = {engine: factory(engine) for engine in plan.engines}
    for engine, backend in backends.items():
        if (reason := backend.unavailable_reason()) is not None:
            raise StillsError(f"{engine} native viewer unavailable: {reason}")
    return backends


def _render_swing(
    plan: ShowcasePlan,
    settings: ExportSettings,
    ready: Mapping[str, NativeBackend],
    job: tuple[str, SwingInput, Mapping[str, int]],
) -> list[dict[str, Any]]:
    from PIL import Image

    from scripts.render_address_engine_stills import annotate

    name, swing, frames = job
    sha = swing.bundle.spec_sha256
    stills_dir = plan.out_dir / name / "stills"
    records: list[dict[str, Any]] = []
    for engine, backend in ready.items():
        indices = [frames[e] for e in EVENTS]
        rendered = list(backend.render(swing, settings, indices, None))
        if len(rendered) != len(EVENTS):
            raise StillsError(f"{engine} returned {len(rendered)} of 3 frames")
        for event, images in zip(EVENTS, rendered, strict=True):
            for view in settings.views:
                frame = np.asarray(images[view])
                if frame.ndim != 3 or not frame.any():
                    raise StillsError(f"{engine} {view} {event}: blank frame")
                time_s = swing.time_s(frames[event])
                label = [
                    f"{engine} / {view}: {event} (v2 trunk visuals)",
                    f"frame {frames[event]}  t = {time_s:.3f} s  spec {sha[:SHA_CHARS]}",
                ]
                path = (
                    stills_dir / f"{name}_{engine}_{view}_{event}_{sha[:SHA_CHARS]}.png"
                )
                stills_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(annotate(frame, label)).save(path)
                records.append(
                    {
                        "file": path.relative_to(plan.out_dir).as_posix(),
                        "swing": name,
                        "engine": engine,
                        "view": view,
                        "event": event,
                        "frame": frames[event],
                        "time_s": time_s,
                        "spec_sha256": sha,
                    }
                )
    return records


def _write_manifest(plan: ShowcasePlan, body: Mapping[str, Any]) -> Path:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture": plan.capture,
        "command": plan.command,
        **body,
    }
    path = plan.out_dir / f"v2_trunk_stills_manifest_{plan.capture}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def render_v2_stills(
    plan: ShowcasePlan,
    settings: ExportSettings,
    *,
    backend_factory: Callable[[str], NativeBackend] | None = None,
    event_finder: Callable[[SwingInput], Mapping[str, int]] | None = None,
    bundle_loader: Callable[[Path], InputBundle] | None = None,
) -> Path:
    """Render address, top and impact stills for every swing and engine.

    Every swing bundle must carry the ``plan.v2_spec`` document (``ValueError``).
    Each PNG is named and labelled with spec sha, engine, view, event and frame;
    the manifest lists them with the command. ``StillsError`` when a viewer
    cannot render or a frame is blank.
    """
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.backends.registry import make_backend
    from src.tools.native_viewer_export.core import SwingInput

    if plan.v2_spec is None:
        raise ValueError("render_v2_stills needs plan.v2_spec")
    expected = load_v2_spec(plan.v2_spec)
    ready = _ready_backends(plan, backend_factory or make_backend)
    records: list[dict[str, Any]] = []
    for name, path in plan.swings.items():
        bundle = (bundle_loader or InputBundle.load)(path)
        if json.loads(bundle.spec_bytes) != expected:
            raise ValueError(
                f"swing {name!r}: the bundle does not carry the --v2-spec document"
            )
        swing = SwingInput(
            bundle,
            bundle.reference_q,
            f"{plan.capture}_{name}",
            name,
            bundle.reference_engine,
        )
        frames = _checked_frames(
            (event_finder or detect_event_frames)(swing), len(bundle.reference_q)
        )
        records += _render_swing(plan, settings, ready, (name, swing, frames))

    return _write_manifest(plan, {"stills": records})
