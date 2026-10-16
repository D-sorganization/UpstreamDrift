"""R8 (#12685) evidence: MuJoCo address stills from the shared body meshes.

Renders the canonical capture-A driver and capture-B 7-iron MuJoCo swings
(spec + forward-dynamics ``q`` from the private swing datastore) at address,
face-on and down-the-line, with the DEFAULT appearance document (charcoal
trousers), at the same camera as the R1 stills (``../trunk_cross_section``),
which are the "before" row. It also writes ``trunk_and_outfit.json``: the
trunk/pelvis sizing (source, ML semi-axis, AP/ML aspect) and the garment
material and coverage of the pelvis and every leg body, from
``model_appearance.body_meshes``.

Usage, from the repository root (Windows uses a hidden GLFW context; Linux
sets ``MUJOCO_GL=egl``)::

    PYTHONPATH=. python3 docs/development/full_body_models/evidence/appearance/\
shared_appearance_spec/render_shared_appearance_spec.py --datastore D:/FleetData/\
swing-datastore --out OUT_DIR [--evidence EVIDENCE_DIR]

Stills are visual evidence, not a pixel unit test; appearance never feeds
dynamics.
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


def _outfit_summary(spec: dict[str, Any], doc: Any) -> dict[str, Any]:
    from src.shared.python.model_appearance import library
    from src.shared.python.model_appearance.body_meshes import (
        body_meshes,
        section_size,
    )

    trunk = {}
    garments: dict[str, str] = {}
    for item in body_meshes(spec, doc):
        part = library.classify_body(item.body)
        if part in ("torso", "pelvis") and item.label == "skin0":
            size = section_size(spec, item.body)
            trunk[item.body.rsplit("/", 1)[-1]] = {
                "source": size.source,
                "ml_semi_axis_m": round(size.radius_m, 4),
                "ap_over_ml": round(size.aspect, 4),
            }
        if part in ("pelvis", "thigh", "shin") and item.label.startswith("garment"):
            garments[item.body.rsplit("/", 1)[-1]] = item.material_name
    return {
        "clothing": doc.clothing,
        "trunk": trunk,
        "lower_garments": garments,
        "thigh_coverage": list(library.garment_coverage(doc.clothing, "thigh")),
        "shin_coverage": list(library.garment_coverage(doc.clothing, "shin")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datastore", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl" if os.name != "nt" else "glfw")
    import imageio.v2 as imageio
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.golf_view_presets import mujoco_camera_params
    from src.shared.python.model_appearance import document_from_dict

    args.out.mkdir(parents=True, exist_ok=True)
    doc = document_from_dict({"schema_version": "appearance-v1"})
    report: dict[str, Any] = {"camera_distance_m": DISTANCE_M, "swings": {}}
    for swing, rel in SWINGS.items():
        folder = args.datastore / "swings" / rel
        raw = (folder / "full_body_spec_hipcal_scaled.json").read_bytes()
        spec = json.loads(raw)
        rec = np.load(folder / "dynamics_record.npz")
        model = mujoco.MjModel.from_xml_string(
            export_full_body_mjcf(raw, appearance=doc)[0]
        )
        data = mujoco.MjData(model)
        adr = [model.joint(n).qposadr[0] for n in spec["coordinate_order"]]
        names = [model.body(i).name for i in range(model.nbody)]
        chest = model.body(next(n for n in names if n.endswith("COMRod"))).id
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
        try:
            data.qpos[adr] = rec["q"][0]
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
                name = f"r8_{swing}_{view}_address.png"
                imageio.imwrite(args.out / name, image)
                if args.evidence is not None:
                    imageio.imwrite(args.evidence / name, image)
        finally:
            renderer.close()
        report["swings"][swing] = _outfit_summary(spec, doc)
    target = args.evidence or args.out
    (target / "trunk_and_outfit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
