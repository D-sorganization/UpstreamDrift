"""Colour per-segment MuJoCo skin pieces by axial load (#15150).

``apply_mujoco_scene_colors`` recolours scene geoms. A skinned body is drawn
from MuJoCo skins, which are not scene geoms: a skin takes its colour from its
model material when the frame is rendered. A skin that is cut into one piece
per segment, each with its own untextured material, can therefore be shaded by
writing that material's colour before the scene is updated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from .axial_loads import AxialLoadFrame
from .force_colors import ForceColorScale

__all__ = ["SkinPiece", "apply_mujoco_skin_colors", "skin_piece_bindings"]

Rgba = tuple[float, float, float, float]


@dataclass(frozen=True)
class SkinPiece:
    """One skin piece: its material, the body whose load colours it, base RGBA."""

    material: str
    body: str
    base_rgba: Rgba


def _check_model(model: Any) -> Any:
    import mujoco

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be a native MuJoCo MjModel")
    return mujoco


def skin_piece_bindings(
    model: Any, piece_bodies: Mapping[str, str]
) -> tuple[SkinPiece, ...]:
    """Bind each named skin to the body whose axial load colours it.

    Preconditions: every key names a skin of ``model`` that has a material,
    every value names a body, and no material serves two bodies (``ValueError``
    otherwise; the colour of a shared material could not follow both loads).
    Postcondition: one ``SkinPiece`` per key, in key order, holding the
    material colour at the time of the call as the base colour.
    """
    mujoco = _check_model(model)
    pieces: list[SkinPiece] = []
    owner: dict[str, str] = {}
    for skin_name, body in piece_bodies.items():
        skin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SKIN, skin_name)
        if skin < 0:
            raise ValueError(f"unknown skin {skin_name!r}")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body) < 0:
            raise ValueError(f"unknown body {body!r} for skin {skin_name!r}")
        material_id = int(model.skin_matid[skin])
        if material_id < 0:
            raise ValueError(f"skin {skin_name!r} has no material to recolour")
        material = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
        if owner.setdefault(material, body) != body:
            raise ValueError(
                f"material {material!r} is shared by bodies {owner[material]!r} "
                f"and {body!r}"
            )
        red, green, blue, alpha = (float(c) for c in model.mat_rgba[material_id])
        pieces.append(SkinPiece(material, body, (red, green, blue, alpha)))
    return tuple(pieces)


def _check_alpha(skin_alpha: float) -> float:
    if isinstance(skin_alpha, bool) or not isinstance(skin_alpha, Real):
        raise TypeError("skin_alpha must be a real number")
    if not math.isfinite(skin_alpha) or not 0.0 <= skin_alpha <= 1.0:
        raise ValueError("skin_alpha must be finite and within [0, 1]")
    return float(skin_alpha)


def apply_mujoco_skin_colors(
    model: Any,
    pieces: Sequence[SkinPiece],
    frame: AxialLoadFrame | None,
    scale: ForceColorScale,
    *,
    skin_alpha: float = 1.0,
) -> int:
    """Write each piece's material colour for ``frame``; call before update_scene.

    A piece whose body has a load takes ``scale.shade_rgb`` over its base
    colour. A piece without one (scale off, no frame, missing or non-finite
    load) is reset to its base colour, because a model material keeps its last
    colour, unlike a scene geom. ``skin_alpha`` multiplies every piece's base
    opacity, so vectors inside the body can be read through the skin.

    Preconditions: ``pieces`` come from :func:`skin_piece_bindings` of this
    ``model`` (``ValueError`` for an unknown material). Postcondition: returns
    the number of pieces shaded by a load; only ``model.mat_rgba`` changes.
    """
    mujoco = _check_model(model)
    if not isinstance(scale, ForceColorScale):
        raise TypeError("scale must be ForceColorScale")
    if frame is not None and not isinstance(frame, AxialLoadFrame):
        raise TypeError("frame must be AxialLoadFrame or None")
    alpha = _check_alpha(skin_alpha)
    shaded = 0
    for piece in pieces:
        material = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, piece.material
        )
        if material < 0:
            raise ValueError(f"unknown material {piece.material!r}")
        base = piece.base_rgba
        load = None if frame is None else frame.values_n.get(piece.body)
        rgb = scale.shade_rgb(load, base[:3])
        shaded += rgb is not None
        model.mat_rgba[material, :3] = base[:3] if rgb is None else rgb
        model.mat_rgba[material, 3] = base[3] * alpha
    return shaded
