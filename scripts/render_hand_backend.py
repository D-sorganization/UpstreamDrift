#!/usr/bin/env python3
"""Render registered hand backends on the club and write review evidence.

Writes one PNG per backend, hand, view and shaft roll, an optional both-hands
frame, an optional contact sheet per hand, and a manifest naming the engine
and the settings hash (#15150). Example::

    PYTHONPATH=. python3 scripts/render_hand_backend.py \\
        --backend capsule --views palm,back,thumb_side,pinky_side,butt,eye \\
        --rolls -20,0,20 --both-hands --contact-sheet \\
        --out docs/development/full_body_models/evidence/hand_render

The manifest is ``render_manifest.json`` by default. A file with ``receipt``
in its name under an evidence root is swept into the matched-swing ledger,
which this picture evidence is not part of; pass ``--manifest-name`` to
choose another name deliberately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.model_appearance.geometry import Mesh  # noqa: E402
from src.shared.python.model_appearance.hand_backends import render  # noqa: E402
from src.shared.python.model_appearance.hand_backends.render_rig import (  # noqa: E402
    result_basis,
    result_meshes,
    result_rgba,
)

DEFAULT_OUT = REPO_ROOT / "docs/development/full_body_models/evidence/hand_render"
DEFAULT_MANIFEST = "render_manifest.json"
SIDES = ("lead", "trail")


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _roll_tag(roll: float) -> str:
    return f"roll{'m' if roll < 0 else 'p'}{abs(round(roll)):02d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, frame: Any, **extra: Any) -> dict[str, Any]:
    entry = {"file": path.name, "sha256": _sha256(path), "engine": frame.engine}
    fill = getattr(frame, "fill_fraction", None)
    if fill is not None:
        entry["fill_fraction"] = round(float(fill), 4)
    entry.update(extra)
    return entry


def both_hands_parts(lead: Any, trail: Any, settings: Any) -> list[render.ScenePart]:
    """Both hands on one club, in the lead hand's grip frame.

    The trail hand is moved toward the head by the difference of the two
    ``distance_below_butt_m`` values, which both results must carry.
    """
    offset = float(trail.distance_below_butt_m) - float(lead.distance_below_butt_m)
    parts, heights = [], []
    for result, shift in ((lead, 0.0), (trail, offset)):
        rgba = result_rgba(result, settings)
        for mesh in result_meshes(result):
            moved = Mesh(mesh.vertices + np.array([0.0, 0.0, shift]), mesh.faces)
            parts.append(render.ScenePart(moved, rgba, "skin"))
            heights.append(moved.vertices[:, 2])
    z = np.concatenate(heights)
    radius = getattr(lead, "grip_radius_m", None)
    club = render.build_club_parts(
        float(z.min()), float(z.max()), settings=settings, grip_radius_m=radius
    )
    return [*parts, *club]


def _contact_sheets(
    args: argparse.Namespace,
    sides: list[str],
    rolls: list[float],
    settings: Any,
    out: Path,
) -> list[dict[str, Any]]:
    """Write one contact sheet per side when ``args.contact_sheet`` is set."""
    sheets: list[dict[str, Any]] = []
    if not args.contact_sheet:
        return sheets
    for side in sides:
        sheet = render.render_contact_sheet(
            _csv(args.backend),
            rolls,
            view=args.sheet_view,
            side=side,
            settings=settings,
            engine=args.engine,
        )
        path = out / f"contact_sheet_{side}_{args.sheet_view}.png"
        render.save_png(sheet, path)
        sheets.append(
            _record(path, sheet, side=side, view=args.sheet_view, rolls_deg=rolls)
        )
    return sheets


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Render everything ``args`` asks for and return the manifest."""
    settings = render.load_hand_render_settings(args.settings)
    if args.frame_px is not None:
        settings = settings.with_overrides(frame_px=args.frame_px)
    views = _csv(args.views) if args.views else list(settings.views)
    rolls = (
        [float(r) for r in _csv(args.rolls)] if args.rolls else list(settings.rolls_deg)
    )
    sides = _csv(args.sides)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend, gl_errors = (
        (None, {}) if args.engine == "matplotlib" else render.gl_engine_status(settings)
    )

    frames: list[dict[str, Any]] = []
    for name in _csv(args.backend):
        results = {
            side: render.build_render_source(
                name, side, grip_radius_m=settings.grip_radius_m
            )
            for side in sides
        }
        for side, result in results.items():
            for view in views:
                for roll in rolls:
                    frame = render.render_view(
                        result,
                        view,
                        shaft_roll_deg=roll,
                        side=side,
                        settings=settings,
                        engine=args.engine,
                    )
                    path = out / f"{name}_{side}_{view}_{_roll_tag(roll)}.png"
                    render.save_png(frame, path)
                    frames.append(
                        _record(
                            path,
                            frame,
                            backend=name,
                            side=side,
                            view=view,
                            roll_deg=roll,
                        )
                    )
        if args.both_hands and set(SIDES) <= set(results):
            parts = both_hands_parts(results["lead"], results["trail"], settings)
            basis = result_basis(results["lead"], settings=settings)
            for view in views:
                frame = render.render_scene(
                    parts, view, basis=basis, settings=settings, engine=args.engine
                )
                path = out / f"{name}_both_{view}_{_roll_tag(0.0)}.png"
                render.save_png(frame, path)
                frames.append(
                    _record(
                        path, frame, backend=name, side="both", view=view, roll_deg=0.0
                    )
                )
    sheets = _contact_sheets(args, sides, rolls, settings, out)

    engines = sorted({entry["engine"] for entry in frames})
    manifest = {
        "issue": 15150,
        "kind": "hand_render_evidence",
        "engines": engines,
        "quality_bar_evidence": engines != ["matplotlib"] and bool(engines),
        "gl_backend": backend,
        "gl_errors": gl_errors,
        "settings_file": "scripts/config/hand_render.json"
        if args.settings is None
        else Path(args.settings).name,
        "settings_hash": settings.settings_hash,
        "frame_px": settings.frame_px,
        "fill_fraction_target": settings.fill_fraction,
        "frames": frames,
        "contact_sheets": sheets,
    }
    review = out / "self_review.json"
    if review.is_file():
        manifest["self_review"] = json.loads(review.read_text(encoding="utf-8"))
    (out / args.manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--backend", default="capsule", help="comma-separated sources")
    parser.add_argument(
        "--views", default="", help="comma-separated views (default: all)"
    )
    parser.add_argument(
        "--rolls", default="", help="comma-separated shaft rolls in degrees"
    )
    parser.add_argument("--sides", default=",".join(SIDES), help="lead, trail or both")
    parser.add_argument(
        "--both-hands", action="store_true", help="add a both-hands frame"
    )
    parser.add_argument(
        "--contact-sheet", action="store_true", help="write contact sheets"
    )
    parser.add_argument(
        "--sheet-view", default="palm", help="view of the contact sheet"
    )
    parser.add_argument(
        "--engine", default=None, help="force osmesa, egl or matplotlib"
    )
    parser.add_argument("--frame-px", type=int, default=None, help="override frame_px")
    parser.add_argument(
        "--settings", default=None, help="settings JSON (default: repo file)"
    )
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    merged: list[str] = []
    for item in raw:
        # "--rolls -20,0,20": argparse would read the value as an option.
        if merged and merged[-1] == "--rolls":
            merged[-1] = f"--rolls={item}"
        else:
            merged.append(item)
    args = build_parser().parse_args(merged)
    manifest = run(args)
    print(
        f"wrote {len(manifest['frames'])} frame(s) and "
        f"{len(manifest['contact_sheets'])} sheet(s) to {args.out} "
        f"with engine(s) {manifest['engines']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
