"""R9a (#12688) evidence: Drake and Pinocchio MeshCat stills with the shared appearance.

Renders capture-A driver at address and impact, face-on and down-the-line,
through the native-viewer backends (Drake MeshCat, Pinocchio MeshCat, and
MuJoCo for the side-by-side row). Uses the DEFAULT appearance document (R8).

Swing data may be supplied either as a same-input ``.npz`` bundle (as in R9b)
or as a MuJoCo canonical-seeds artifact folder under the private swing
datastore (``full_body_spec_hipcal_scaled.json`` + ``dynamics_record.npz``).

Usage, from the repository root (Linux brick with pydrake, pinocchio, coal,
meshcat, playwright, and Chromium; Windows can render MuJoCo only)::

    PYTHONPATH=.:src python3 \\
docs/development/full_body_models/evidence/appearance/engines/drake_pinocchio/\\
render_drake_pinocchio_appearance.py --datastore D:/FleetData/swing-datastore \\
--out OUT_DIR [--engines drake,pinocchio,mujoco]

Stills are visual evidence, not a pixel unit test; appearance never feeds
dynamics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ENGINES_DEFAULT = ("drake", "pinocchio", "mujoco")
VIEWS = ("face_on", "down_the_line")
WIDTH, HEIGHT = 960, 720
CAPTURE_A_DRIVER = (
    "swings/capA-driver__mujoco__canonical-seeds/"
    "2026-10-05_best_matches/artifacts/anthro_driver_seeds"
)


def phase_indices(bundle: object) -> dict[str, int]:
    """Address (first sample) and impact (bundle impact split) sample indices."""
    provenance = getattr(bundle, "provenance", {}) or {}
    impact_s = provenance.get("impact_split_time_s")
    if impact_s is None:
        raise ValueError("bundle provenance has no impact_split_time_s")
    last = int(np.asarray(bundle.reference_q).shape[0]) - 1  # type: ignore[attr-defined]
    impact = min(last, int(round(float(impact_s) / float(bundle.dt_s))))  # type: ignore[attr-defined]
    return {"address": 0, "impact": impact}


def swing_from_datastore(folder: Path, *, swing: str, club: str) -> Any:
    """Build a :class:`SwingInput` from a canonical-seeds artifact folder."""
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import SwingInput

    spec_path = folder / "full_body_spec_hipcal_scaled.json"
    record_path = folder / "dynamics_record.npz"
    if not spec_path.is_file() or not record_path.is_file():
        raise FileNotFoundError(f"missing spec or dynamics_record under {folder}")
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    rec = np.load(record_path)
    q = np.asarray(rec["q_track"], dtype=float)
    names = tuple(spec["coordinate_order"])
    if q.shape[1] != len(names):
        raise ValueError("q_track width must match spec coordinate_order")
    times = np.asarray(rec["track_time_s"], dtype=float)
    if times.shape[0] != q.shape[0]:
        raise ValueError("track_time_s length must match q_track rows")
    dt_s = float(np.median(np.diff(times)))
    impact_s = None
    receipt_path = folder / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        impact_s = ((receipt.get("ik") or {}).get("impact_split") or {}).get(
            "impact_time_s"
        )
    if impact_s is None:
        raise ValueError(f"{receipt_path} has no impact_split.impact_time_s")
    bundle = InputBundle(
        spec_bytes=spec_bytes,
        coordinate_order=names,
        dt_s=dt_s,
        q0=q[0],
        v0=np.zeros(len(names)),
        efforts=np.zeros((len(q) - 1, len(names))),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
        provenance={"impact_split_time_s": float(impact_s)},
    )
    return SwingInput(bundle, q, swing, club, "mujoco")


def bundle_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compose_row(
    out: Path,
    *,
    label: str,
    phase: str,
    view: str,
    engine_paths: dict[str, Path],
) -> Path:
    """Horizontal strip: MeshCat engines then MuJoCo (lighting differs by engine)."""
    import imageio.v2 as imageio

    order = [e for e in ("drake", "pinocchio", "mujoco") if e in engine_paths]
    if len(order) < 2:
        raise ValueError("need at least two engine stills to compose a comparison row")
    tiles = [imageio.imread(engine_paths[e]) for e in order]
    heights = {t.shape[0] for t in tiles}
    if len(heights) != 1:
        raise ValueError("all tiles in a row must share height")
    combined = np.concatenate(tiles, axis=1)
    path = out / f"r9a_driver_{label}_{view}_{phase}.png"
    imageio.imwrite(path, combined)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="same-input bundle .npz")
    parser.add_argument(
        "--datastore",
        type=Path,
        help="swing datastore root (uses capture-A driver canonical seeds)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engines", default=",".join(ENGINES_DEFAULT))
    parser.add_argument("--swing", default="capture-A")
    parser.add_argument("--club", default="Driver")
    args = parser.parse_args()
    import imageio.v2 as imageio

    from src.tools.native_viewer_export.backends.registry import make_backend
    from src.tools.native_viewer_export.core import ExportSettings, load_swing_input

    if args.bundle is not None:
        swing = load_swing_input(args.bundle, swing=args.swing, club=args.club)
        bundle_digest = bundle_sha256(args.bundle)
        bundle_source = str(args.bundle)
    elif args.datastore is not None:
        folder = args.datastore / CAPTURE_A_DRIVER
        swing = swing_from_datastore(folder, swing=args.swing, club=args.club)
        bundle_digest = hashlib.sha256(swing.bundle.spec_bytes).hexdigest()
        bundle_source = str(folder)
    else:
        raise SystemExit("pass --bundle or --datastore")

    phases = phase_indices(swing.bundle)
    settings = ExportSettings(
        views=VIEWS,
        width=WIDTH,
        height=HEIGHT,
        hud=False,
        overlays=False,
        ball=False,
        multiview=False,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rendered: dict[tuple[str, str, str], Path] = {}
    for engine in args.engines.split(","):
        engine = engine.strip()
        if not engine:
            continue
        backend = make_backend(engine)
        reason = backend.unavailable_reason()
        if reason is not None:
            raise RuntimeError(f"{engine} backend unavailable: {reason}")
        frames = backend.render(swing, settings, list(phases.values()), None)
        for phase, views in zip(phases, frames, strict=True):
            for view, image in views.items():
                path = args.out / f"r9a_driver_{engine}_{view}_{phase}.png"
                imageio.imwrite(path, np.asarray(image))
                rendered[(engine, view, phase)] = path
                print(path)  # noqa: T201 - evidence script progress

    meshcat = {k: v for k, v in rendered.items() if k[0] in {"drake", "pinocchio"}}
    if "mujoco" in args.engines and meshcat:
        for view in VIEWS:
            for phase in phases:
                engines_present = {
                    e: rendered[(e, view, phase)]
                    for e in ("drake", "pinocchio", "mujoco")
                    if (e, view, phase) in rendered
                }
                if len(engines_present) >= 2:
                    row = compose_row(
                        args.out,
                        label="meshcat_mujoco_compare",
                        phase=phase,
                        view=view,
                        engine_paths=engines_present,
                    )
                    print(row)  # noqa: T201

    receipt = {
        "issue": "#12688",
        "epic": "#12678",
        "bundle_sha256": bundle_digest,
        "bundle_source": bundle_source,
        "swing": {
            "capture": args.swing,
            "club": args.club,
            "frames": phases,
            "dt_s": float(swing.bundle.dt_s),
            "impact_split_time_s": float(
                (swing.bundle.provenance or {}).get("impact_split_time_s", 0.0)
            ),
        },
        "settings": {
            "size": [WIDTH, HEIGHT],
            "views": list(VIEWS),
            "hud": False,
            "overlays": False,
            "ball": False,
        },
        "command": (
            "PYTHONPATH=.:src python3 "
            "docs/development/full_body_models/evidence/appearance/engines/"
            "drake_pinocchio/render_drake_pinocchio_appearance.py "
            f"--datastore {args.datastore or '<datastore>'} --out <dir>"
        ),
    }
    receipt_path = args.out / "r9a_appearance_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt_path)  # noqa: T201


if __name__ == "__main__":
    main()
