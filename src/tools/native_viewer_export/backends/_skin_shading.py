"""Axial-load shading of a skinned body in the MuJoCo render worker (#15150).

A skinned body is drawn from MuJoCo skins, which the scene-geom shading never
reaches. When segment shading is on, the worker therefore cuts every skin of
the exported MJCF into one piece per segment (``body_skin_pieces``), binds
each piece to the body whose load colours it, with the load alias of
``skin_load_settings.json``, and writes the piece colours before each scene
update (``mujoco_skin_colors``). Hair, eyes, brows and lips keep their colour.

With shading off, or a model without a skin, the MJCF string is returned as
it came and nothing is bound. Display only.

A piece's material has no texture, while the skin's own is drawn multiplied
by its texture. Each piece's base colour is therefore multiplied once by that
texture's mean colour, so an unloaded piece keeps the tone of the uncut skin
(checked on the appearance layer's flat cube textures, without their marks).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.mujoco_force_colors import unshaded_geom_ids
from src.shared.python.body_part_viz.mujoco_skin_colors import (
    SkinPiece,
    apply_mujoco_skin_colors,
    skin_piece_bindings,
)
from src.shared.python.body_part_viz.skin_load_settings import (
    SkinLoadSettings,
    aliased_load_bodies,
    load_skin_load_settings,
)

_SKIN_TAG = "<skin"
_WHITE = (1.0, 1.0, 1.0)

AxialFrames = Sequence[AxialLoadFrame | None]


@dataclass(frozen=True)
class SkinShading:
    """What the worker shades a skinned model with; empty for any other model."""

    pieces: tuple[SkinPiece, ...] = ()
    keep_geoms: frozenset[int] = frozenset()

    def colour(
        self, model: Any, frame: AxialLoadFrame | None, scale: ForceColorScale
    ) -> int:
        """Write the piece colours of ``frame``; call before ``update_scene``.

        Postcondition: returns the number of pieces shaded by a load; without
        pieces the model is not touched and the result is 0. Raises as
        ``apply_mujoco_skin_colors`` does.
        """
        if not self.pieces:
            return 0
        return apply_mujoco_skin_colors(model, self.pieces, frame, scale)


def skin_pieces_wanted(
    axial_frames: AxialFrames | None, shading: ForceColorScale | None
) -> bool:
    """True when this export shades segments, so a body skin has to be cut.

    False without staged loads, without a scale, for a disabled scale and for
    one of zero opacity (which draws exactly as shading off). ``TypeError``
    for a ``shading`` that is neither a ``ForceColorScale`` nor ``None``.
    """
    if shading is not None and not isinstance(shading, ForceColorScale):
        raise TypeError("shading must be a ForceColorScale or None")
    if axial_frames is None or shading is None:
        return False
    return shading.enabled and shading.opacity > 0.0


def split_scene_skins(xml: str, wanted: bool) -> tuple[str, dict[str, str]]:
    """The MJCF with each skin cut per segment, and piece skin name -> body.

    Preconditions: ``xml`` is an MJCF string (``TypeError`` otherwise).
    Postcondition: unless ``wanted`` and the MJCF has a skin, the result is
    ``xml`` itself and an empty table; else as ``split_mjcf_skins``.
    """
    if not isinstance(xml, str):
        raise TypeError("xml must be an MJCF string")
    if not wanted or _SKIN_TAG not in xml:
        return xml, {}
    from src.engines.physics_engines.mujoco.python.body_skin_pieces import (
        split_mjcf_skins,
    )

    return split_mjcf_skins(xml)


def _native(model: Any) -> Any:
    import mujoco

    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be a native MuJoCo MjModel")
    return mujoco


def texture_tone(model: Any, material: int) -> tuple[float, float, float]:
    """Mean RGB (0..1) of the colour texture of material id ``material``.

    White for a material without one. Preconditions: ``model`` is a native
    MjModel (``TypeError``) and ``material`` one of its material ids
    (``ValueError``).
    """
    mujoco = _native(model)
    if not 0 <= material < model.nmat:
        raise ValueError(f"the model has no material {material}")
    textures = model.mat_texid[material]
    texture = int(textures[mujoco.mjtTextureRole.mjTEXROLE_RGB])
    if texture < 0:
        return _WHITE
    channels = int(model.tex_nchannel[texture])
    start = int(model.tex_adr[texture])
    count = int(model.tex_width[texture]) * int(model.tex_height[texture]) * channels
    pixels = model.tex_data[start : start + count].reshape(-1, channels)
    red, green, blue = (float(c) for c in pixels[:, :3].mean(axis=0) / 255.0)
    return red, green, blue


def _restore_texture_tones(model: Any, piece_bodies: Mapping[str, str]) -> None:
    """Multiply each piece material by the mean of its skin material's texture."""
    mujoco = _native(model)
    for piece in piece_bodies:
        skin = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SKIN, piece)
        if skin < 0:
            raise ValueError(f"unknown skin {piece!r}")
        own = int(model.skin_matid[skin])
        if own < 0:
            continue  # skin_piece_bindings reports the missing material
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, own)
        # split_skins: piece <skin>__seg<n> wears <material>__seg__<skin>__<n>
        parent, _, index = piece.rpartition("__seg")
        source = name.removesuffix(f"__seg__{parent}__{index}")
        worn = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, source)
        if worn < 0 or worn == own:
            continue
        model.mat_rgba[own, :3] *= texture_tone(model, worn)


def bind_skin_shading(
    model: Any,
    piece_bodies: Mapping[str, str],
    settings: SkinLoadSettings | None = None,
) -> SkinShading:
    """Bind the skin pieces of ``model`` and find the geoms that stay unshaded.

    ``piece_bodies`` is the table of :func:`split_scene_skins`; ``settings``
    default to ``skin_load_settings.json``. An alias to a body the model does
    not have is left out, so the piece follows its own body.

    Preconditions: ``model`` is a native MjModel compiled from the cut MJCF
    and ``settings`` a ``SkinLoadSettings`` (``TypeError``); every piece is a
    skin of the model with a material (``ValueError``). Call once per model.
    Postcondition: an empty ``SkinShading`` for an empty table, and the model
    is then unchanged; else one ``SkinPiece`` per piece, whose base colour is
    the piece material's colour times its texture tone (written to
    ``model.mat_rgba``), and the ids of the geoms wearing an unshaded material.
    """
    mujoco = _native(model)
    if not isinstance(piece_bodies, Mapping):
        raise TypeError("piece_bodies must map piece skin names to body names")
    if settings is not None and not isinstance(settings, SkinLoadSettings):
        raise TypeError("settings must be SkinLoadSettings or None")
    if not piece_bodies:
        return SkinShading()
    chosen = settings or load_skin_load_settings()
    alias = {
        body: load
        for body, load in chosen.load_body_alias.items()
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, load) >= 0
    }
    _restore_texture_tones(model, piece_bodies)
    pieces = skin_piece_bindings(model, aliased_load_bodies(piece_bodies, alias))
    keep = unshaded_geom_ids(model, chosen.unshaded_material_prefixes)
    return SkinShading(pieces, keep)


def compile_shaded_model(
    xml: str,
    axial_frames: AxialFrames | None,
    shading: ForceColorScale | None,
    settings: SkinLoadSettings | None = None,
) -> tuple[Any, SkinShading]:
    """Compile the worker's model, with its skins cut and bound when shaded.

    Postcondition: with shading off (:func:`skin_pieces_wanted`) or no skin in
    ``xml``, the model is compiled from ``xml`` itself and the ``SkinShading``
    is empty; else as :func:`split_scene_skins` and :func:`bind_skin_shading`.
    """
    import mujoco

    wanted = skin_pieces_wanted(axial_frames, shading)
    cut, piece_bodies = split_scene_skins(xml, wanted)
    model = mujoco.MjModel.from_xml_string(cut)
    return model, bind_skin_shading(model, piece_bodies, settings)
