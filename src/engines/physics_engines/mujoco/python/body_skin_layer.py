"""MuJoCo translation of the skinned CC0 body: three native skins (#15299).

With ``subject.body_visual_backend`` set to ``skinned_cc0`` the appearance
layer leaves out the loft meshes of trunk, arms and legs and this module adds
one ``<deformable><skin>`` per layer (skin, shirt, trousers) instead. Each
skin vertex is a weighted blend of the model's own rigid bodies, so the
surface stays continuous over every joint.

The stored skin is fitted to the subject's joint centres, which are read from
the model itself at the bind pose (``export_settings.json``: arms hanging),
dressed, and bound to the bodies (``model_appearance.body_skin_export``).
A skin is drawn only: no geom, no mass, no contact; ``nq``, the body tree,
masses and inertias are those of the loft model. A MuJoCo skin has no texture
coordinates and ignores its material's texture while meshes are multiplied by
it, so the skin material is written without one: forearm, neck, hands and
face then share one tone.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET  # nosec B405 # nosemgrep: python.lang.security.use-defused-xml.use-defused-xml - construction only; parsing is defused
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance import body_skin_dress as dress
from src.shared.python.model_appearance import body_skin_export as export
from src.shared.python.model_appearance import body_skin_fit as fit
from src.shared.python.model_appearance import body_skin_pose as pose
from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.schema import AppearanceDocument

#: Layer -> the role whose garment (or bare skin) material it wears.
_WEARER = {"skin": None, "shirt": "thorax", "trousers": "thigh_l"}
_TEXTURE_KEYS = ("texture", "texrepeat", "texuniform")


def _nums(values: Any, digits: int) -> str:
    return " ".join(format(float(v), f".{digits}g") for v in np.ravel(values))


def reference_frames(
    spec: Mapping[str, Any], coordinates: Mapping[str, float]
) -> dict[str, pose.Rigid]:
    """World pose of every model body at ``coordinates`` (others zero).

    Compiles the plain (no visuals) export of ``spec``; raises ``ValueError``
    for a coordinate the model does not have.
    """
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    plain = {**spec, "subject": dict(spec.get("subject", {}))}
    plain["subject"].pop("body_visual_backend", None)
    model = mujoco.MjModel.from_xml_string(
        export_full_body_mjcf(json.dumps(plain).encode())[0]
    )
    data = mujoco.MjData(model)
    for name, value in coordinates.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"the model has no coordinate {name!r}")
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    return {
        model.body(index).name: pose.Rigid(
            data.xmat[index].reshape(3, 3).copy(), data.xpos[index].copy()
        )
        for index in range(1, model.nbody)
    }


def _layer_material(
    doc: AppearanceDocument, roles: Mapping[str, str], layer: str
) -> str:
    wearer = _WEARER[layer]
    garment = None if wearer is None else library.garment_name_for(doc, roles[wearer])
    return garment or library.material_name_for(doc, roles["pelvis"])


def _untextured(root: ET.Element, name: str) -> None:
    for material in root.iter("material"):
        if material.get("name") == name:
            for key in _TEXTURE_KEYS:
                material.attrib.pop(key, None)


def attach_body_skin(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    material_for: Callable[[str], str],
    doc: AppearanceDocument,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Add the skin, shirt and trousers skins to ``root``; returns a summary.

    ``elements`` are the exporter's MJCF bodies by spec name and
    ``material_for`` registers a library material and returns its MJCF name.
    Postcondition: only ``<deformable>`` and material assets are added.
    """
    settings, posing = export.load_export_settings(), pose.load_pose_settings()
    by_body = reference_frames(spec, settings.bind_pose)
    roles = export.body_roles(spec, {b: f.translation for b, f in by_body.items()})
    frames = {role: by_body[body] for role, body in roles.items()}
    skin = fit.load_body_skin()
    profile = spec.get("subject", {}).get("body_skin_profile")
    fitted = fit.fit_body_skin(
        skin, export.subject_lengths(frames), fit.load_fit_settings(profile=profile)
    )
    weights = pose.smooth_shoulder_weights(skin, fitted, posing)
    layers = dress.dress_skin(skin, fitted, weights, dress.load_dress_settings())
    binds = export.bind_poses(frames, fitted)
    shares = export.group_shares(skin.groups, posing, settings)
    deformable = ET.SubElement(root, "deformable")
    summary: dict[str, Any] = {"bodies": roles, "bones": {}, "vertices": {}}
    for name, layer in layers.items():
        material = material_for(_layer_material(doc, roles, name))
        if name == "skin":
            _untextured(root, material)
        node = ET.SubElement(
            deformable,
            "skin",
            name=f"body_skin_{name}",
            material=material,
            vertex=_nums(layer.vertices, 6),
            face=" ".join(str(int(i)) for i in layer.faces.ravel()),
        )
        bones = export.skin_bones(layer.weights, skin.groups, shares, binds, settings)
        for bone in bones:
            x, y, z, w = Rotation.from_matrix(bone.bind.rotation).as_quat()
            ET.SubElement(
                node,
                "bone",
                body=str(elements[roles[bone.role]].get("name")),
                bindpos=_nums(bone.bind.translation, 9),
                bindquat=_nums((w, x, y, z), 9),
                vertid=" ".join(str(int(i)) for i in bone.vertex_ids),
                vertweight=_nums(bone.weights, 5),
            )
        summary["bones"][name] = len(bones)
        summary["vertices"][name] = len(layer.vertices)
    return summary
