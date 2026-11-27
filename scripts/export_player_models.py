"""Export and render source-attributed player appearance candidates.

Example: ``MUJOCO_GL=egl PYTHONPATH=.:src python scripts/export_player_models.py --output DIR``.
Review poses are hash-bound kinematic inputs, never captured swing motion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

if sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

PLAYERS = ("tiger_woods", "ben_hogan")
SIZE_DESKTOP = (1280, 960)
SIZE_PHONE = (390, 844)


def _bounds(model: Any, data: Any, mujoco: Any, indices: Any) -> tuple[Any, Any]:
    """Return exact geometry bounds for meshes and conservative sphere bounds."""
    import numpy as np

    points = []
    for i in indices:
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(model.geom_dataid[i])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            points.append(
                model.mesh_vert[start : start + count]
                @ data.geom_xmat[i].reshape(3, 3).T
                + data.geom_xpos[i]
            )
        else:
            center, radius = data.geom_xpos[i], float(model.geom_rbound[i])
            points.append(np.stack((center - radius, center + radius)))
    if not points:
        raise ValueError("Cannot compute bounds for an empty geometry selection")
    stacked = np.concatenate(points)
    return stacked.min(axis=0), stacked.max(axis=0)


def _ground_shoes(model: Any, data: Any, mujoco: Any) -> float:
    """Shift the lowest sole vertex to z=0 for display, without contact claims."""
    soles = []
    for i in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        body = (model.body(int(model.geom_bodyid[i])).name or "").lower()
        if "sole" not in name or not any(part in body for part in ("calcn", "toes")):
            continue
        mesh = int(model.geom_dataid[i])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        soles.append(
            float(
                (
                    model.mesh_vert[start : start + count]
                    @ data.geom_xmat[i].reshape(3, 3).T
                    + data.geom_xpos[i]
                )[:, 2].min()
            )
        )
    if not soles:
        raise ValueError("Could not find named shoe-sole meshes to ground preview")
    shift = -min(soles)
    for name in ("TranslationInputZ", "PelvisTz"):
        try:
            address = int(model.joint(name).qposadr[0])
        except KeyError:
            continue
        data.qpos[address] += shift
        mujoco.mj_forward(model, data)
        return shift
    raise ValueError("No named root vertical translation joint found")


def _validated_review_pose(
    pose: dict[str, Any], spec: dict[str, Any], xml_bytes: bytes
) -> dict[str, float]:
    """Verify review-pose hashes and return finite q values keyed by coordinate."""
    import numpy as np

    from src.shared.python.model_appearance.schema import physics_spec_sha256

    spec_hash = physics_spec_sha256(spec)
    xml_hash = hashlib.sha256(xml_bytes).hexdigest()
    if pose.get("target_physics_spec_sha256") != spec_hash:
        raise ValueError("review pose does not match the model physics spec hash")
    if pose.get("target_model_xml_sha256") != xml_hash:
        raise ValueError("review pose does not match the model XML hash")
    names, q = pose.get("coordinate_order"), pose.get("q")
    spec_names = spec.get("coordinate_order")
    if not isinstance(names, list) or not isinstance(spec_names, list):
        raise ValueError("review pose and model need coordinate_order lists")
    if len(names) != len(set(names)) or set(names) != set(spec_names):
        raise ValueError("review-pose coordinates must exactly match model coordinates")
    values = np.asarray(q, dtype=float)
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("review-pose q must contain one finite value per coordinate")
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _load_scene(
    xml_path: Path, spec: dict[str, Any], coordinates: dict[str, float] | None
) -> tuple[Any, Any, Any, Any, Any, float, float, float]:
    """Load an MJCF scene, optionally apply a verified pose, and measure fit bounds."""
    import mujoco
    import numpy as np

    from src.shared.python.model_appearance.head import (
        head_frame_in_body,
        resolve_head_anchor,
    )
    from src.shared.python.model_appearance.schema import load_appearance
    from src.shared.python.model_appearance.skinned_body import spec_body_poses

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if coordinates is not None:
        for name, value in coordinates.items():
            try:
                address = int(model.joint(name).qposadr[0])
            except KeyError as exc:
                raise ValueError(
                    f"review-pose coordinate {name!r} is absent from MJCF"
                ) from exc
            data.qpos[address] = value
    mujoco.mj_forward(model, data)
    grounding = _ground_shoes(model, data, mujoco)
    all_geoms = np.flatnonzero(model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    bodies = {
        int(i): (model.body(int(model.geom_bodyid[i])).name or "").lower()
        for i in all_geoms
    }
    names = {
        int(i): (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or ""
        ).lower()
        for i in all_geoms
    }
    props = [
        i
        for i in all_geoms
        if any(p in bodies[int(i)] for p in ("/club", "/shaft", "clubface"))
        and not names[int(i)].endswith(("_hand", "_grip"))
    ]
    helpers = [
        i
        for i in all_geoms
        if names[int(i)].startswith(("visual_com_", "visual_frame_"))
    ]
    person = [
        int(i)
        for i in all_geoms
        if int(i) not in helpers
        and (int(i) not in props or names[int(i)].endswith(("_hand", "_grip")))
    ]
    head = [i for i in person if names[i].startswith("visual_head_")]
    if not person or not head:
        raise ValueError(f"MJCF has no named golfer or visual head geoms: {xml_path}")
    anchor = resolve_head_anchor(spec)
    if anchor is None:
        raise ValueError("player spec has no head anchor for front-view camera")
    head_pose = spec_body_poses(spec, coordinates)[anchor.body] @ head_frame_in_body(
        anchor, load_appearance(xml_path.with_name("model.appearance.json")).head
    )
    forward = head_pose[:3, 0]
    if np.linalg.norm(forward[:2]) < 1e-6:
        raise ValueError("head forward axis has no horizontal projection")
    front_azimuth = math.degrees(math.atan2(-forward[1], -forward[0])) % 360
    fit_low, fit_high = _bounds(model, data, mujoco, np.asarray([*person, *props]))
    body_low, body_high = _bounds(model, data, mujoco, np.asarray(person))
    return (
        model,
        data,
        (person, props, helpers, head),
        (fit_low, fit_high),
        grounding,
        float((body_high - body_low)[2]),
        float((fit_high - fit_low)[2]),
        front_azimuth,
    )


def _common_camera_distance(items: list[dict[str, Any]]) -> float:
    """Fit every selected player and pose to one shared full-body camera scale."""
    extent = 0.0
    for item in items:
        variants = (
            [None] if item["pose_values"] is None else [None, item["pose_values"]]
        )
        for coordinates in variants:
            scene = _load_scene(item["xml"], item["spec"], coordinates)
            extent = max(extent, scene[-2])
    return max(3.6, extent * 1.65)


def _render(
    xml: Path,
    out: Path,
    spec: dict[str, Any],
    coordinates: dict[str, float] | None,
    camera_distance: float,
) -> dict[str, Any]:
    """Render front, profile, head, and phone views at the shared camera scale."""
    import imageio.v3 as iio
    import mujoco
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    model, data, groups, fit, grounding, height, _extent, front_azimuth = _load_scene(
        xml, spec, coordinates
    )
    person, props, helpers, head = groups
    low, high = fit
    model.geom_rgba[np.asarray([*props, *helpers], dtype=int), 3] = 0
    head_low, head_high = _bounds(model, data, mujoco, np.asarray(head, dtype=int))
    center = (low + high) / 2
    views = {
        "front": (front_azimuth, center, camera_distance),
        "profile": ((front_azimuth + 90.0) % 360, center, camera_distance),
        "head": (
            front_azimuth,
            (head_low + head_high) / 2,
            max((head_high - head_low).max() * 2.5, 0.35),
        ),
    }
    model.vis.global_.offwidth, model.vis.global_.offheight = SIZE_DESKTOP
    images: dict[str, str] = {}
    for width, height_px in (SIZE_DESKTOP, SIZE_PHONE):
        renderer = mujoco.Renderer(model, height_px, width)
        try:
            for label, (azimuth, lookat, distance) in views.items():
                if width == SIZE_PHONE[0] and label != "front":
                    continue
                model.geom_rgba[np.asarray(props, dtype=int), 3] = (
                    0 if label == "head" else 1
                )
                camera = mujoco.MjvCamera()
                camera.lookat[:] = lookat
                camera.distance, camera.azimuth, camera.elevation = (
                    distance,
                    azimuth,
                    -10.0,
                )
                renderer.update_scene(data, camera=camera)
                path = out / f"{label}_{width}x{height_px}.png"
                iio.imwrite(path, renderer.render().copy())
                images[label + ("_phone" if width == SIZE_PHONE[0] else "")] = str(path)
        finally:
            renderer.close()
    return {
        "previews": images,
        "grounding_translation_m": grounding,
        "grounding_method": "lowest shoe-sole vertex to z=0; not a two-foot-contact claim",
        "rendered_geometry_height_m": height,
        "pose": "validated review coordinates"
        if coordinates is not None
        else "model default qpos",
        "motion_qualification": "not established by this renderer",
    }


def _selected_players(
    requested: tuple[str, ...], catalog: tuple[str, ...]
) -> tuple[str, ...]:
    """Expand catalog selections in order, preserving the legacy two-player alias."""
    if not requested:
        raise ValueError("Select at least one player")
    selected: dict[str, None] = {}
    for item in requested:
        expanded = catalog if item == "all" else PLAYERS if item == "both" else (item,)
        for player in expanded:
            if player not in catalog:
                raise ValueError(f"Unknown player {player!r}")
            selected[player] = None
    return tuple(selected)


def main(argv: list[str] | None = None) -> int:
    """Export or render selected player models with optional validated review poses."""
    from src.shared.python.model_appearance.player_models import (
        export_player,
        player_ids,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    catalog = player_ids()
    parser.add_argument(
        "--player", nargs="+", choices=(*catalog, "both", "all"), default=("both",)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-spec", type=Path, help="Export appearance on a base physics spec"
    )
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Render existing output/player models",
    )
    parser.add_argument(
        "--pose-root", type=Path, help="Directory containing player/review_pose.json"
    )
    args = parser.parse_args(argv)
    if args.render_existing and args.base_spec:
        parser.error("--base-spec cannot be used with --render-existing")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = _selected_players(tuple(args.player), catalog)
    items: list[dict[str, Any]] = []
    for player in selected:
        folder = args.output / player
        spec_path = folder / "model.json"
        if args.render_existing:
            xml = folder / "model.xml"
            if not xml.is_file() or not spec_path.is_file():
                raise FileNotFoundError(
                    f"existing model files are missing under {folder}"
                )
        else:
            xml = export_player(player, folder, base_spec=args.base_spec)
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        pose, values = None, None
        if args.pose_root:
            pose_path = args.pose_root / player / "review_pose.json"
            if not pose_path.is_file():
                raise FileNotFoundError(
                    f"review pose missing for {player}: {pose_path}"
                )
            pose = json.loads(pose_path.read_text(encoding="utf-8"))
            values = _validated_review_pose(pose, spec, xml.read_bytes())
        items.append(
            {
                "player": player,
                "folder": folder,
                "xml": xml,
                "spec": spec,
                "pose": pose,
                "pose_values": values,
            }
        )
    camera_distance = _common_camera_distance(items)
    for item in items:
        folder = item["folder"]
        modes = ("neutral", "address") if item["pose"] is not None else ("neutral",)
        renders = {}
        for mode in modes:
            coords = item["pose_values"] if mode == "address" else None
            receipt = _render(
                item["xml"],
                folder / f"previews/{mode}",
                item["spec"],
                coords,
                camera_distance,
            )
            if mode == "address":
                receipt["review_pose"] = {
                    k: item["pose"].get(k)
                    for k in (
                        "method",
                        "status",
                        "diagnostics",
                        "closure_pass",
                        "closure_threshold_m",
                        "closure_threshold_deg",
                    )
                }
                receipt["review_pose_source"] = f"{item['player']}/review_pose.json"
                receipt["hashes_matched"] = True
            receipt["camera_distance_m"] = camera_distance
            receipt["previews"] = {
                k: str(Path(v).relative_to(folder))
                for k, v in receipt["previews"].items()
            }
            renders[mode] = receipt
            (folder / f"previews/{mode}/preview_receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
