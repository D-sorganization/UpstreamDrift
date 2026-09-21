"""View source-bound full-body candidate poses without running dynamics (MV-02, #10478).

The source model has inertial solids rather than surface meshes. Decoupled visual skins
(inertia ellipsoids vs. anatomical meshes) and multi-layer rendering provide cosmetic
visualization without modifying model physics, contact laws, or coordinate trajectories.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.body_part_viz.anatomical_visuals import (
    VisualSkinMode,
    resolve_anatomical_visual,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """Validated kinematic trajectory matched to a qualified model specification."""

    specification: dict[str, Any]
    names: tuple[str, ...]
    time_s: np.ndarray
    q: np.ndarray
    model_sha256: str


def _validate_candidate_timestamps(
    timestamps: np.ndarray, num_names: int, q: np.ndarray
) -> None:
    """Assert increasing finite physical time and matching trajectory dimensions."""
    if (
        timestamps.ndim != 1
        or len(timestamps) < 2
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
        or q.shape != (len(timestamps), num_names)
        or not np.isfinite(q).all()
    ):
        raise ValueError(
            "Candidate requires finite configurations and strictly increasing physical time"
        )


def load_candidate(candidate: Path, specification: Path) -> Candidate:
    """Require exact model identity and map configurations by coordinate name."""
    raw = specification.read_bytes()
    spec = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    receipt_path = candidate.with_name("receipt.json")
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_bytes())
        if receipt.get("document_sha256") != digest:
            raise ValueError(
                "Candidate/model hash mismatch or missing model provenance"
            )

    names = tuple(spec["coordinate_order"])
    with np.load(candidate, allow_pickle=False) as data:
        source = tuple(str(x) for x in data["coordinate_order"])
        q = np.asarray(data["q"], dtype=float)
        timestamps = np.asarray(data["time_s"], dtype=float)

    if len(set(source)) != len(source) or set(source) != set(names):
        raise ValueError("Candidate coordinate names must uniquely match the model")

    _validate_candidate_timestamps(timestamps, len(names), q)
    ordered_q = q[:, [source.index(n) for n in names]]
    return Candidate(spec, names, timestamps, ordered_q, digest)


def _build_solid_ellipsoid(
    solid: dict[str, Any],
    joint: int,
    body_pose: Any,
    geometry: Any,
    pin: Any,
    coal: Any,
) -> None:
    """Compute and attach a uniform-density inertia ellipsoid to a native joint."""
    mass = float(solid["mass_kg"])
    if mass <= 0:
        return
    inertia = np.asarray(solid["inertia_com_kg_m2"], dtype=float)
    covariance = (np.trace(inertia) / 2 * np.eye(3) - inertia) / mass
    eigenvalues, basis = np.linalg.eigh(covariance)
    if np.min(eigenvalues) < -1e-9:
        raise ValueError(f"Nonphysical visual inertia for {solid['name']}")
    if np.linalg.det(basis) < 0:
        basis[:, 0] *= -1
    radii = np.maximum(np.sqrt(5 * np.maximum(eigenvalues, 0)), 0.004)
    pose = body_pose * pin.SE3(np.asarray(solid["placement"], dtype=float))
    pose = pose * pin.SE3(basis, np.asarray(solid["com_m"], dtype=float))
    shape = coal.Ellipsoid(*radii.tolist())
    obj = pin.GeometryObject(f"solid_{geometry.ngeoms}", joint, pose, shape)
    label = solid["name"].lower()
    obj.meshColor = np.array(
        [0.95, 0.6, 0.12, 1.0] if "club" in label else [0.18, 0.55, 0.8, 1.0]
    )
    geometry.addGeometryObject(obj)


def _attach_anatomical_solid(
    geometry: Any,
    joint: int,
    body_pose: Any,
    solid: dict[str, Any],
    body_name: str,
    asset_root: Path | None,
    pin: Any,
    coal: Any,
) -> None:
    """Attach anatomical mesh or visible diagnostic fallback for a body solid."""
    binding = resolve_anatomical_visual(body_name, solid.get("name"))
    if binding is None:
        return
    placement = solid.get("placement")
    solid_placement = (
        pin.SE3(np.asarray(placement, dtype=float))
        if placement is not None
        else pin.SE3.Identity()
    )
    pose = body_pose * solid_placement
    resolved_file: Path | None = None
    if asset_root is not None:
        candidate_mesh = asset_root / Path(binding.mesh_relative_path).name
        if candidate_mesh.is_file():
            resolved_file = candidate_mesh
    if resolved_file is None:
        direct = Path(binding.mesh_relative_path)
        if direct.is_file():
            resolved_file = direct

    if resolved_file is not None and not binding.is_fallback:
        try:
            scale_vec = np.asarray(binding.scale, dtype=float)
            obj = pin.GeometryObject(
                f"visual_{geometry.ngeoms}_{binding.semantic_body}",
                joint,
                pose,
                str(resolved_file),
                scale_vec,
            )
            obj.meshColor = np.asarray(binding.color, dtype=float)
            geometry.addGeometryObject(obj)
            return
        except Exception as exc:
            logger.warning(
                "Failed to load mesh %s; using diagnostic fallback: %s",
                resolved_file,
                exc,
            )

    # Visible diagnostic fallback
    shape = (
        coal.Box(*binding.scale)
        if binding.fallback_geometry_type == "box"
        else coal.Sphere(max(binding.scale) / 2)
    )
    obj = pin.GeometryObject(
        f"visual_{geometry.ngeoms}_{binding.semantic_body}",
        joint,
        pose,
        shape,
    )
    obj.meshColor = np.asarray(binding.color, dtype=float)
    geometry.addGeometryObject(obj)


def _attach_contact_spheres(geometry: Any, plant: Any, pin: Any, coal: Any) -> None:
    """Add physical contact spheres as a translucent collision layer."""
    for sphere in plant.contact_spheres:
        joint, body_pose = plant._bodies[sphere.body]
        offset_pose = pin.SE3(np.eye(3), np.asarray(sphere.position_m, dtype=float))
        pose = body_pose * offset_pose
        shape = coal.Sphere(float(sphere.radius_m))
        obj = pin.GeometryObject(f"collision_contact_{sphere.name}", joint, pose, shape)
        obj.meshColor = np.array([0.1, 0.9, 0.2, 0.5])
        geometry.addGeometryObject(obj)


def build_visuals(
    plant: Any,
    specification: dict[str, Any],
    skin_mode: str = "anatomical_mesh",
    layers: tuple[str, ...] = ("visual",),
    asset_root: Path | None = None,
) -> Any:
    """Attach visual skins to the exact native body frames without altering model physics."""
    import coal

    pin: Any = import_module("pinocchio")
    geometry = pin.GeometryModel()

    for body in specification.get("bodies", []):
        if body["name"] == "world":
            continue
        joint, body_pose = plant._bodies[body["name"]]
        for solid in body.get("solids", []):
            if (
                "inertia" in layers
                or skin_mode == VisualSkinMode.INERTIA_ELLIPSOIDS.value
            ):
                _build_solid_ellipsoid(solid, joint, body_pose, geometry, pin, coal)
            elif "visual" in layers and skin_mode in (
                VisualSkinMode.ANATOMICAL_MESH.value,
                VisualSkinMode.ANATOMICAL_CAPSULE.value,
            ):
                _attach_anatomical_solid(
                    geometry,
                    joint,
                    body_pose,
                    solid,
                    body["name"],
                    asset_root,
                    pin,
                    coal,
                )

    if "collision" in layers:
        _attach_contact_spheres(geometry, plant, pin, coal)

    return geometry


def launch_native_candidate(
    candidate_path: Path,
    spec_path: Path,
    viewer: str,
    output: Path | None = None,
    *,
    fps: float = 30.0,
    speed: float = 1.0,
    loop: bool = False,
    skin_mode: str = "anatomical_mesh",
) -> None:
    """Validate presentation options and launch native candidate playback."""
    import math

    from .viewer_presentation import gepetto_playback_lock

    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(speed) or speed <= 0:
        raise ValueError("Presentation fps and speed must be finite and positive")
    if viewer == "gepetto":
        with gepetto_playback_lock():
            _launch_native_candidate(
                candidate_path, spec_path, viewer, output, fps, speed, loop, skin_mode
            )
    else:
        _launch_native_candidate(
            candidate_path, spec_path, viewer, output, fps, speed, loop, skin_mode
        )


def _launch_meshcat_player(
    plant: Any,
    visual: Any,
    configurations: list[Any],
    candidate: Candidate,
    output: Path | None,
    speed: float,
    loop: bool,
    title: str,
) -> None:
    """Render trajectory animation in MeshCat and export static HTML if requested."""
    import meshcat.geometry as geom
    from meshcat.animation import Animation
    from pinocchio.visualize import MeshcatVisualizer

    pin: Any = import_module("pinocchio")
    viz = MeshcatVisualizer(plant.model, pin.GeometryModel(), visual)
    viz.initViewer(open=False)
    viz.loadViewerModel(title)
    viz.viewer["/Background"].set_property("top_color", [0.94, 0.96, 0.99])
    viz.viewer["/Background"].set_property("bottom_color", [0.78, 0.83, 0.89])
    viz.viewer["ground"].set_object(
        geom.Box([5, 5, 0.012]), geom.MeshLambertMaterial(color=0xCBD5D8)
    )
    ground = np.eye(4)
    ground[2, 3] = (
        float(candidate.specification["contact"]["ground"]["height_m"] or 0) - 0.006
    )
    viz.viewer["ground"].set_transform(ground)
    frame_rate = 1.0 / float(np.median(np.diff(candidate.time_s)))
    animation = Animation(default_framerate=frame_rate * speed)
    for k, q in enumerate(configurations):
        pin.forwardKinematics(plant.model, viz.data, q)
        pin.updateGeometryPlacements(plant.model, viz.data, visual, viz.visual_data)
        with animation.at_frame(viz.viewer, k) as frame:
            for i, obj in enumerate(visual.geometryObjects):
                node = viz.getViewerNodeName(obj, pin.GeometryType.VISUAL)
                frame[node].set_transform(viz.visual_data.oMg[i].homogeneous)
    viz.display(configurations[0])
    viz.viewer.set_animation(animation, play=False, repetitions=0 if loop else 1)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        banner = (
            '<div style="position:fixed;left:12px;top:12px;padding:12px;'
            "background:#ffffffed;color:#17324d;border-radius:8px;"
            'font:14px sans-serif;pointer-events:none;z-index:10">'
            "<b>Pinocchio — Full-Swing Fitted Motion</b><br>"
            "Kinematic playback · Not accepted torque-driven dynamics<br>"
            "Anatomical visual skin · Native Z-up<br>"
            "Open Controls → Animations → default: play, pause, time, timeScale"
            "</div>"
        )
        output.write_text(
            viz.viewer.static_html().replace("</body>", banner + "</body>"),
            encoding="utf-8",
        )
    logger.warning("KINEMATIC PLAYBACK — unaccepted dynamics; anatomical visual skin")


def _launch_native_candidate(
    candidate_path: Path,
    spec_path: Path,
    viewer: str,
    output: Path | None,
    fps: float,
    speed: float,
    loop: bool,
    skin_mode: str,
) -> None:
    """Launch the actual native model with saved poses; never label as replay."""
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    candidate = load_candidate(candidate_path, spec_path)
    plant = FullBodyPinocchioModel(candidate.specification)
    visual = build_visuals(plant, candidate.specification, skin_mode=skin_mode)
    configurations = [
        plant.configuration(dict(zip(candidate.names, row, strict=True)))
        for row in candidate.q
    ]
    title = "Pinocchio_Kinematic_Playback_Not_Accepted_Dynamics"
    if viewer == "meshcat":
        _launch_meshcat_player(
            plant, visual, configurations, candidate, output, speed, loop, title
        )
    elif viewer == "gepetto":
        from pinocchio.visualize import GepettoVisualizer

        from .viewer_presentation import presentation_frames

        pin: Any = import_module("pinocchio")
        viz = GepettoVisualizer(plant.model, pin.GeometryModel(), visual)
        viz.initViewer(loadModel=False)
        viz.loadViewerModel(title)
        viz.display(configurations[0])
        while True:
            for k in presentation_frames(candidate.time_s, fps, speed):
                viz.display(configurations[k])
            if not loop:
                return
            time.sleep(0.5)
    else:
        raise ValueError("Native candidate viewer must be meshcat or gepetto")
