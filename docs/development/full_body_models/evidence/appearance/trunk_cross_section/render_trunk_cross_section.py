"""R1 (#12680) evidence: MuJoCo appearance stills and chest silhouette widths.

Renders the canonical capture-A driver and capture-B 7-iron MuJoCo swings
(spec + forward-dynamics ``q`` from the private swing datastore) at address and
impact, face-on and down-the-line, at one camera distance, and measures the
chest (COMRod body: skin and polo) silhouette across the trunk's image-space
long axis from a segmentation pass. ``--mode before`` emulates the pre-fix
width axis (``geometry._frame``'s default body +x) for the comparison row.

Usage, from the repository root (Windows uses a hidden GLFW context; Linux
sets ``MUJOCO_GL=egl``)::

    PYTHONPATH=. python3 docs/development/full_body_models/evidence/appearance/\
trunk_cross_section/render_trunk_cross_section.py --datastore D:/FleetData/\
swing-datastore --out OUT_DIR --mode after [--evidence EVIDENCE_DIR]

Stills are visual evidence, not a pixel unit test; appearance never feeds
dynamics. Impact is the receipt's forward-dynamics impact time when present,
else the frame, within 0.15 s after the peak club-body speed, where the club
body returns closest to its address position (noted per row).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SWINGS = {
    "driver": "capA-driver__mujoco__canonical-seeds/2026-10-05_best_matches/"
    "artifacts/anthro_driver_seeds",
    "iron7": "capB-iron7__mujoco__canonical-seeds-zmp/2026-10-05_best_matches/"
    "artifacts/iron_seeds_zmp",
}
WIDTH, HEIGHT, DISTANCE_M = 960, 720, 3.2
VIEWS = ("face_on", "down_the_line")


def _emulate_pre_fix_width_axis() -> None:
    """Patch the appearance layer back to the old default (+x) width axis."""
    from src.engines.physics_engines.mujoco.python import appearance_layer
    from src.shared.python.model_appearance import geometry

    def old_hint(sections: Any, body: str, start: Any, end: Any) -> Any:
        if body not in sections:
            return None
        return geometry._frame(np.subtract(end, start))[0]

    appearance_layer.width_hint_for = old_hint  # type: ignore[assignment]


def _impact_frame(mujoco: Any, model: Any, data: Any, adr: list, rec: Any, receipt):
    t_imp = receipt["dynamics"].get("fd_phase", {}).get("impact_time_s")
    if t_imp is not None:
        return int(np.argmin(np.abs(rec["time_s"] - t_imp))), "receipt impact_time_s"
    names = [model.body(i).name for i in range(model.nbody)]
    club = model.body(next(n for n in names if n.endswith("Clubface Vector"))).id
    pos = []
    for q in rec["q"]:
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        pos.append(data.xpos[club].copy())
    # Impact proxy: within 0.15 s after the peak club-body speed (downswing),
    # the frame where the club body comes back closest to its address position.
    t = rec["time_s"]
    gap = np.linalg.norm(np.asarray(pos) - pos[0], axis=1)
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
    peak = int(np.argmax(speed)) + 1
    window = np.nonzero((t >= t[peak]) & (t <= t[peak] + 0.15))[0]
    k = int(window[np.argmin(gap[window])])
    return k, "club body closest to its address position after peak speed"


def _silhouette(mask: np.ndarray) -> dict[str, float]:
    """Extent across and along the mask's principal axis (0.5-99.5 percentile)."""
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))

    def span(axis: np.ndarray) -> float:
        proj = pts @ axis
        return round(float(np.percentile(proj, 99.5) - np.percentile(proj, 0.5)), 1)

    return {
        "chest_pixels": int(mask.sum()),
        "chest_width_across_trunk_px": span(vecs[:, 0]),
        "chest_length_along_trunk_px": span(vecs[:, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datastore", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--mode", choices=("before", "after"), default="after")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    import imageio.v2 as imageio
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.golf_view_presets import mujoco_camera_params
    from src.shared.python.model_appearance import document_from_dict

    if args.mode == "before":
        _emulate_pre_fix_width_axis()
    suffix = "_before" if args.mode == "before" else ""
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for swing, rel in SWINGS.items():
        folder = args.datastore / "swings" / rel
        raw = (folder / "full_body_spec_hipcal_scaled.json").read_bytes()
        spec = json.loads(raw)
        rec = np.load(folder / "dynamics_record.npz")
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        doc = document_from_dict({"schema_version": "appearance-v1"})
        model = mujoco.MjModel.from_xml_string(
            export_full_body_mjcf(raw, appearance=doc)[0]
        )
        data = mujoco.MjData(model)
        adr = [model.joint(n).qposadr[0] for n in spec["coordinate_order"]]
        names = [model.body(i).name for i in range(model.nbody)]
        chest = model.body(next(n for n in names if n.endswith("COMRod"))).id
        chest_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == chest]
        k_imp, how = _impact_frame(mujoco, model, data, adr, rec, receipt)
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        try:
            for label, k in (("address", 0), ("impact", k_imp)):
                data.qpos[adr] = rec["q"][k]
                mujoco.mj_forward(model, data)
                for view in VIEWS:
                    p = mujoco_camera_params(view, data.xpos[chest].copy(), DISTANCE_M)
                    cam = mujoco.MjvCamera()
                    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    cam.lookat[:] = p.lookat
                    cam.distance, cam.azimuth = p.distance, p.azimuth
                    cam.elevation = p.elevation
                    renderer.update_scene(data, camera=cam)
                    image = renderer.render().copy()
                    name = f"r1_{swing}_{view}_{label}{suffix}.png"
                    imageio.imwrite(args.out / name, image)
                    if args.evidence is not None and label == "address":
                        imageio.imwrite(args.evidence / name, image)
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera=cam)
                    seg = renderer.render().copy()
                    renderer.disable_segmentation_rendering()
                    mask = np.isin(seg[..., 0], chest_geoms) & (
                        seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                    )
                    rows.append(
                        {
                            "swing": swing,
                            "frame": label,
                            "view": view,
                            "frame_index": k,
                            "time_s": float(rec["time_s"][k]),
                            "impact_source": how,
                            **_silhouette(mask),
                        }
                    )
        finally:
            renderer.close()
    report = {
        "mode": args.mode,
        "camera_distance_m": DISTANCE_M,
        "image_px": [WIDTH, HEIGHT],
        "rows": rows,
    }
    target = args.evidence or args.out
    (target / f"silhouette_widths_{args.mode}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
