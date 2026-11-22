"""MuJoCo translation of the skinned CC0 body: three native skins (#15299).

With ``subject.body_visual_backend`` set to ``skinned_cc0`` the appearance
layer leaves out the loft meshes of trunk, arms and legs and this module adds
one ``<deformable><skin>`` per layer (skin, shirt, trousers) instead. Each
skin vertex is a weighted blend of the model's own rigid bodies, so the
surface stays continuous over every joint. A skinned hand is rigid on the
club; the forearm's end is blended to the body that hand moves with, so the
wrist stays closed under the hand's cuff at any bend (``hand_binds``).

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
from src.shared.python.model_appearance.club_assembly import ClubAssembly
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


def _untextured(root: ET.Element, name: str, rgba: tuple[float, ...]) -> None:
    """Drop the texture of material ``name`` and give its colour back to ``rgba``.

    A textured material is emitted with a white rgba (the texture carries the
    colour once); without the texture the colour must be written to rgba.
    """
    for material in root.iter("material"):
        if material.get("name") == name:
            for key in _TEXTURE_KEYS:
                material.attrib.pop(key, None)
            material.set("rgba", _nums(rgba, 6))


def spec_turns(
    spec: Mapping[str, Any],
    roles: Mapping[str, str],
    frames: Mapping[str, pose.Rigid],
    coordinates: Mapping[str, float],
) -> dict[str, Any]:
    """Per role, the rotation from the model body's frame to the spec body's.

    The exporter re-frames bodies (the trail forearm is half a turn about its
    x axis from the spec's). ``frames`` are the model's at ``coordinates``.
    """
    from src.shared.python.model_appearance.skinned_body import spec_body_poses

    posed = spec_body_poses(spec, dict(coordinates))
    return {
        role: posed[body][:3, :3].T @ frames[role].rotation
        for role, body in roles.items()
    }


def wrist_rings(
    spec: Mapping[str, Any],
    club: ClubAssembly | None,
    roles: Mapping[str, str],
    binds: Mapping[str, pose.Rigid],
    turns: Mapping[str, Any],
) -> dict[str, fit.WristRing]:
    """Each skinned hand's wrist section in the fitted rest space, by side.

    ``turns`` are :func:`spec_turns`: the sections are in the spec's forearm
    frames. Empty unless the spec draws the skinned hands on a club.
    """
    from src.shared.python.model_appearance.skinned_wrist_sections import (
        forearm_wrist_sections,
    )

    sections = forearm_wrist_sections(spec, club)
    rings = {}
    for side, _ in fit.SIDES:
        section = sections.get(roles[f"forearm_{side}"])
        if section is not None:
            forearm = f"forearm_{side}"
            turn = binds[forearm].rotation @ turns[forearm].T
            rings[side] = fit.WristRing(
                section.half_width_m,
                section.half_depth_m,
                turn @ section.width_axis,
                turn @ section.centre_m,
            )
    return rings


def hand_binds(
    spec: Mapping[str, Any],
    club: ClubAssembly | None,
    roles: Mapping[str, str],
    binds: Mapping[str, pose.Rigid],
    turns: Mapping[str, Any],
    joints: Mapping[str, Any],
) -> dict[str, pose.Rigid]:
    """Bind pose, by side, of the body each skinned hand is rigid with.

    In the fitted rest space, on the wrist joint of ``joints``, turned as at
    the straight wrist the forearm's end is shaped at (``turns`` are
    :func:`spec_turns`); the forearm's end is blended to it
    (``export.blend_wrists``). Empty unless the spec draws the skinned hands
    on a club.
    """
    from src.shared.python.model_appearance.skinned_wrist_sections import (
        straight_hand_turns,
    )

    straight = straight_hand_turns(spec, club)
    hands = {}
    for side, _ in fit.SIDES:
        forearm, hand = f"forearm_{side}", f"hand_{side}"
        turn = straight.get(roles[forearm])
        if turn is not None:
            relative = turns[forearm].T @ turn @ turns[hand]
            wrist = np.asarray(joints[f"wrist_{side}"], float)
            hands[side] = pose.Rigid(binds[forearm].rotation @ relative, wrist)
    return hands


def attach_body_skin(
    root: ET.Element,
    elements: Mapping[str, ET.Element],
    material_for: Callable[[str], str],
    doc: AppearanceDocument,
    spec: Mapping[str, Any],
    club: ClubAssembly | None = None,
) -> dict[str, Any]:
    """Add the skin, shirt and trousers skins to ``root``; returns a summary.

    ``elements`` are the exporter's MJCF bodies by spec name and
    ``material_for`` registers a library material and returns its MJCF name;
    with the skinned hands of ``club`` the forearms end in the hands' wrists.
    Postcondition: only ``<deformable>`` and material assets are added.
    """
    settings, posing = export.load_export_settings(), pose.load_pose_settings()
    by_body = reference_frames(spec, settings.bind_pose)
    roles = export.body_roles(spec, {b: f.translation for b, f in by_body.items()})
    frames = {role: by_body[body] for role, body in roles.items()}
    skin = fit.load_body_skin()
    profile = spec.get("subject", {}).get("body_skin_profile")
    fitting = fit.load_fit_settings(profile=profile)
    fitted = fit.fit_body_skin(skin, export.subject_lengths(frames), fitting)
    binds = export.bind_poses(frames, fitted)
    turns = spec_turns(spec, roles, frames, settings.bind_pose)
    rings = wrist_rings(spec, club, roles, binds, turns)
    fitted = fit.match_wrists(skin, fitted, rings, fitting)
    hands = hand_binds(spec, club, roles, binds, turns, fitted.joints)
    weights = pose.smooth_shoulder_weights(skin, fitted, posing)
    layers = dress.dress_skin(skin, fitted, weights, dress.load_dress_settings())
    shares = export.group_shares(skin.groups, posing, settings)
    deformable = ET.SubElement(root, "deformable")
    summary: dict[str, Any] = {"bodies": roles, "bones": {}, "vertices": {}}
    summary["wrists_matched"] = sorted(rings)
    summary["wrists_blended"] = sorted(hands) if settings.wrist_blend_length_m else []
    for name, layer in layers.items():
        library_name = _layer_material(doc, roles, name)
        material = material_for(library_name)
        if name == "skin":
            _untextured(
                root, material, library.library_materials(doc)[library_name].rgba
            )
        node = ET.SubElement(
            deformable,
            "skin",
            name=f"body_skin_{name}",
            material=material,
            vertex=_nums(layer.vertices, 6),
            face=" ".join(str(int(i)) for i in layer.faces.ravel()),
        )
        bones = export.skin_bones(layer.weights, skin.groups, shares, binds, settings)
        bones = export.blend_wrists(bones, layer.vertices, binds, hands, settings)
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


def posed_skin_layers(model: Any, data: Any) -> dict[str, tuple[Any, Any]]:
    """Vertices ``(N, 3)`` and faces ``(F, 3)`` of each skin at the pose in ``data``.

    The surface MuJoCo draws, from its own scene update; no renderer is needed.
    Precondition: ``mj_forward`` has run on ``data``. Raises ``ValueError`` for a
    model without a skin.
    """
    import mujoco

    if model.nskin == 0:
        raise ValueError("the model has no skin")
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 1)
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        mujoco.MjvPerturb(),
        mujoco.MjvCamera(),
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )
    flat = np.array(scene.skinvert, dtype=float).reshape(-1, 3)
    layers = {}
    for index in range(model.nskin):
        skin = model.skin(index)
        first, count = int(scene.skinvertadr[index]), int(scene.skinvertnum[index])
        face, faces = int(skin.faceadr[0]), int(skin.facenum[0])
        triangles = np.array(model.skin_face[face : face + faces], dtype=np.int64)
        layers[skin.name] = (flat[first : first + count], triangles)
    return layers
