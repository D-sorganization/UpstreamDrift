"""Build the spec-driven musculoskeletal model (issue #11617, phase 2).

The skeleton is the exported full-body spec model (exact to the MuJoCo
reference); this module removes the loop-closure weld, the ground-contact forces
and the generic force set, then grafts the 80 Rajagopal-Lai-Uhlrich lower-limb
muscles (see :mod:`musculoskeletal_graft` for the frame mathematics).  The
patella, its joint and the coupler constraints are added back for the quadriceps.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import musculoskeletal_graft as graft
from src.engines.physics_engines.opensim.python.full_body_osim import (
    export_full_body_osim,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    _scale_set,
    resolve_base_model,
)
from src.shared.python.contracts import ensure, require

_REMOVED_SETS = (
    "updMarkerSet",
    "updForceSet",
    "updConstraintSet",
    "updContactGeometrySet",
)
#: Talus scale is not recoverable from a mass centre; use the talus-calcaneus
#: joint-offset ratio (spec over generic), which both models expose.
_TALUS_JOINT = "subtalar"


def _osim() -> Any:
    import opensim

    return opensim


def _vec(v: Any) -> np.ndarray:
    return np.array([v.get(i) for i in range(3)], dtype=float)


def _joint_offset_norm(model: Any, joint_name: str, *, child: bool) -> float:
    joint = model.getJointSet().get(joint_name)
    frame = joint.getChildFrame() if child else joint.getParentFrame()
    return float(
        np.linalg.norm(
            _vec(_osim().PhysicalOffsetFrame.safeDownCast(frame).get_translation())
        )
    )


def _base_mass_centres(base: Any) -> dict[str, np.ndarray]:
    return {
        b.getName(): _vec(b.get_mass_center())
        for b in base.getBodySet()
        if graft.side_of(b.getName())
        and b.getName().rsplit("_", 1)[0] in graft.LEG_BODIES
    }


def derive_scales(spec: dict[str, Any], base: Any) -> dict[str, float]:
    """Per-body scale of the generic Rajagopal model that reproduces the spec leg."""
    spec_com = graft.spec_leg_mass_centres(spec)
    joints = {j["name"]: j for j in spec["joints"]}
    talus: list[float] = []
    for side in graft.SIDES:
        offset = np.asarray(joints[f"subtalar_{side}"]["parent_to_base"], float)[:3, 3]
        generic = _joint_offset_norm(base, f"subtalar_{side}", child=False)
        talus.append(float(np.linalg.norm(offset)) / generic)
    return graft.leg_scale_factors(
        spec_com, _base_mass_centres(base), talus_scale=float(np.mean(talus))
    )


def scaled_base_model(
    base_model_path: str | Path | None, scales: dict[str, float]
) -> Any:
    """The generic Rajagopal model scaled with ``scales`` (initialised)."""
    osim = _osim()
    base = osim.Model(str(resolve_base_model(base_model_path)))
    state = base.initSystem()
    base.scale(state, _scale_set(scales), True)
    base.initSystem()
    return base


def load_spec_skeleton(spec_bytes: bytes) -> Any:
    """Spec model with its weld, contact forces, markers and generic forces removed."""
    osim = _osim()
    xml, _ = export_full_body_osim(spec_bytes)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "full_body.osim"
        path.write_text(xml, encoding="utf-8")
        model = osim.Model(str(path))
    for remove in _REMOVED_SETS:
        getattr(model, remove)().clearAndDestroy()
    return model


def _hip_centre(base: Any, side: str) -> np.ndarray:
    return _vec(
        _osim()
        .PhysicalOffsetFrame.safeDownCast(
            base.getJointSet().get(f"hip_{side}").getParentFrame()
        )
        .get_translation()
    )


def _body_of(frame: Any) -> str:
    return str(frame.findBaseFrame().getName())


def _graft_wrap_objects(spec_model: Any, base: Any, ctx: dict[str, Any]) -> int:
    osim = _osim()
    copied = 0
    for body in base.getBodySet():
        wraps = body.getWrapObjectSet()
        for i in range(wraps.getSize()):
            wrap = wraps.get(i)
            clone = wrap.clone()
            target_name = body.getName()
            if target_name == graft.BASE_PELVIS_BODY:
                side = graft.side_of(wrap.getName())
                target_name = graft.SPEC_PELVIS_BODY
                mapped = graft.map_pelvis_point(
                    _vec(wrap.get_translation()),
                    ctx["rotation"],
                    ctx["base_centre"][side],
                    ctx["spec_centre"][side],
                )
                clone.set_translation(osim.Vec3(*[float(x) for x in mapped]))
                angles = graft.map_wrap_orientation(
                    _vec(wrap.get_xyz_body_rotation()), ctx["rotation"]
                )
                clone.set_xyz_body_rotation(osim.Vec3(*[float(x) for x in angles]))
            spec_model.updBodySet().get(target_name).addWrapObject(clone)
            copied += 1
    return copied


def _graft_muscles(spec_model: Any, base: Any, ctx: dict[str, Any]) -> int:
    osim = _osim()
    count = 0
    for muscle in base.getMuscles():
        clone = muscle.clone()
        path = clone.updGeometryPath()
        points = path.getPathPointSet()
        original = muscle.getGeometryPath().getPathPointSet()
        spec_pts = []
        for k in range(original.getSize()):
            point = osim.PathPoint.safeDownCast(original.get(k))
            body = _body_of(point.getParentFrame())
            location = _vec(point.get_location())
            if body == graft.BASE_PELVIS_BODY:
                side = graft.side_of(muscle.getName())
                location = graft.map_pelvis_point(
                    location,
                    ctx["rotation"],
                    ctx["base_centre"][side],
                    ctx["spec_centre"][side],
                )
                body = graft.SPEC_PELVIS_BODY
            spec_pts.append((point.getName(), body, location))
        points.clearAndDestroy()
        for name, body, location in spec_pts:
            path.appendNewPathPoint(
                name,
                spec_model.getBodySet().get(body),
                osim.Vec3(*[float(x) for x in location]),
            )
        clone.set_ignore_tendon_compliance(True)
        spec_model.addForce(clone)
        count += 1
    return count


def build_spec_musculoskeletal_model(
    spec_bytes: bytes, base_model_path: str | Path | None = None
) -> tuple[Any, dict[str, Any]]:
    """Return ``(model, info)``: spec skeleton plus grafted lower-limb muscles.

    Postconditions: the model has the spec's 44 coordinates (plus the dependent
    ``knee_angle_*_beta``), 80 rigid-tendon muscles and no constraints other than
    the patellofemoral couplers.
    """
    osim = _osim()
    spec = graft.spec_document(spec_bytes)
    skeleton = load_spec_skeleton(spec_bytes)
    base = osim.Model(str(resolve_base_model(base_model_path)))
    base.initSystem()
    scales = derive_scales(spec, base)
    base = scaled_base_model(base_model_path, scales)
    rotation, spec_centre, residual = graft.pelvis_frame_from_spec(spec)
    scale = scales[graft.BASE_PELVIS_BODY]
    ctx = {
        "rotation": rotation,
        "spec_centre": spec_centre,
        "base_centre": {s: _hip_centre(base, s) for s in graft.SIDES},
    }
    # Patella, patellofemoral joints and the knee coupler constraints.
    for side in graft.SIDES:
        skeleton.addBody(base.getBodySet().get(f"patella_{side}").clone())
        skeleton.addJoint(base.getJointSet().get(f"patellofemoral_{side}").clone())
    for k in range(base.getConstraintSet().getSize()):
        skeleton.addConstraint(base.getConstraintSet().get(k).clone())
    n_wrap = _graft_wrap_objects(skeleton, base, ctx)
    n_muscle = _graft_muscles(skeleton, base, ctx)
    skeleton.setName("spec_musculoskeletal")
    skeleton.finalizeConnections()
    skeleton.initSystem()
    info = {
        "scales": scales,
        "pelvis_scale": scale,
        "pelvis_axes_residual": residual,
        "n_muscles": n_muscle,
        "n_wrap_objects": n_wrap,
        "n_coordinates": int(skeleton.getCoordinateSet().getSize()),
        "base_model": str(resolve_base_model(base_model_path)),
    }
    ensure(n_muscle > 0, "grafted model has no muscles")
    return skeleton, info
