"""Shared grip-hand renderer: the public facade (#15150).

Any hand backend hands its result to :func:`render_view` and gets back a
solid, smooth-shaded, auto-framed picture of that hand on the club. The
camera rig, lights and every number come from
``scripts/config/hand_render.json``.

Engine order: MuJoCo offscreen through OSMesa, then EGL, then a matplotlib
fallback. The frame says which engine drew it; ``engine == "matplotlib"`` is
not quality-bar evidence. pyrender is not wired in.

Accepted ``result`` shapes, all in the hand-grip frame (grip axis ``z``,
butt toward head): an object with ``.mesh`` (``vertices``, ``faces``) and an
optional ``.skin_rgb``, ``.material_rgba``, ``.material`` (library name) or
``.grip_radius_m``; a bare mesh; or a ``dict`` of meshes.

Shaft roll: the camera basis is taken from the hand before the roll, so a
roll shows as the hand turning about the shaft. Pass ``shaft_roll_deg=0``
when the backend already baked the roll.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.model_appearance.grip_hand_mesh import build_grip_hand_mesh
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends import (
    render_rig as _rig,
)
from src.shared.python.model_appearance.hand_backends.render_matplotlib import (
    render_matplotlib,
)
from src.shared.python.model_appearance.hand_backends.render_mujoco import (
    GLUnavailableError,
    probe_gl_backend,
    render_mujoco,
)
from src.shared.python.model_appearance.hand_backends.render_rig import (
    AnatomicalBasis,
    CameraRig,
    RenderedView,
    ScenePart,
    assemble_scene,
    build_club_parts,
    camera_rig,
    coerce_basis,
    projected_fill,
    result_basis,
    side_name,
    focus_vertices_of,
    skin_vertices_of,
    validated_part,
)
from src.shared.python.model_appearance.hand_backends.render_settings import (
    FALLBACK_ENGINE,
    GL_ENGINES,
    PART_KINDS,
    REQUIRED_VIEWS,
    HandRenderSettings,
    load_hand_render_settings,
    resolve_settings,
)

__all__ = [
    "AnatomicalBasis",
    "CameraRig",
    "ContactSheet",
    "GLUnavailableError",
    "HandRenderSettings",
    "PART_KINDS",
    "REQUIRED_VIEWS",
    "RenderedView",
    "ScenePart",
    "anatomical_basis",
    "assemble_scene",
    "build_club_parts",
    "build_render_source",
    "camera_rig",
    "gl_engine_status",
    "load_hand_render_settings",
    "register_render_source",
    "render_contact_sheet",
    "render_scene",
    "render_sources",
    "render_view",
    "save_png",
]

logger = logging.getLogger(__name__)

#: Builders the contact sheet can call: ``builder(side=..., grip_radius_m=...)``.
_RENDER_SOURCES: dict[str, Callable[..., Any]] = {}


@dataclass(frozen=True)
class ContactSheet:
    """A labelled grid of frames: rows are sources, columns are shaft rolls."""

    image: NDArray[np.uint8]
    engine: str
    cells: tuple[tuple[RenderedView, ...], ...]
    row_labels: tuple[str, ...]
    column_labels: tuple[str, ...]


def anatomical_basis(
    skin_vertices: Any, *, side: Any, settings: HandRenderSettings | None = None
) -> AnatomicalBasis:
    """Default ``(thumb, dorsum, shaft)`` basis from skin vertices.

    A heuristic for backends that do not know their own axes; see
    :func:`render_rig.anatomical_basis`. Raises ``ValueError`` for an empty or
    non-finite vertex array or an unknown ``side``.
    """
    return _rig.anatomical_basis(
        skin_vertices, side=side, settings=resolve_settings(settings)
    )


def gl_engine_status(
    settings: HandRenderSettings | None = None,
) -> tuple[str | None, dict[str, str]]:
    """``(backend, errors)``: the MuJoCo GL backend that works here, if any."""
    return probe_gl_backend(resolve_settings(settings))


def _measured_fill(kind_mask: NDArray[np.uint8], window: Any = None) -> float:
    """Longer side of the skin pixels' bounds over the frame side.

    ``window`` is ``None`` or the ``(lo, hi)`` projected bounds of the focus
    vertices in normalised device coordinates; skin pixels outside it (a
    forearm stub the camera does not frame on) are left out of the measure.
    """
    skin = kind_mask == PART_KINDS.index("skin") + 1
    if window is not None:
        side = kind_mask.shape[0]
        lo, hi = (np.asarray(bound, dtype=float) for bound in window)
        left, right = (np.array([lo[0], hi[0]]) + 1.0) / 2.0 * side
        top, bottom = (1.0 - np.array([hi[1], lo[1]])) / 2.0 * side
        inside = np.zeros_like(skin)
        inside[
            max(0, math.floor(top) - 1) : math.ceil(bottom) + 1,
            max(0, math.floor(left) - 1) : math.ceil(right) + 1,
        ] = True
        skin &= inside
    rows = np.flatnonzero(skin.any(axis=1))
    cols = np.flatnonzero(skin.any(axis=0))
    if rows.size == 0:
        return 0.0
    height = rows[-1] - rows[0] + 1
    width = cols[-1] - cols[0] + 1
    return float(max(height, width)) / float(kind_mask.shape[0])


def render_scene(
    parts: Sequence[ScenePart],
    view: str,
    *,
    basis: Any = None,
    settings: HandRenderSettings | None = None,
    side: Any = "lead",
    engine: str | None = None,
) -> RenderedView:
    """Render positioned parts from one named view.

    ``parts`` is a sequence of :class:`ScenePart` already placed in a common
    hand-grip frame (for example two skin meshes and the club for the
    both-hands frame). The camera auto-frames on the skin, or on the ``focus``
    vertices of a skin part that marks them, so ``settings.fill_fraction`` is
    the share of the frame the hand itself spans. ``basis``
    is ``None`` (vertex heuristic, signed by ``side``), an
    :class:`AnatomicalBasis`, or ``(thumb, dorsum[, shaft])`` vectors.
    ``engine`` forces ``"matplotlib"`` or one GL backend; ``None`` follows
    ``settings.engine_order``.

    Raises ``ValueError`` / ``TypeError`` on a bad part, view, basis or engine.
    """
    settings = resolve_settings(settings)
    if isinstance(parts, ScenePart) or not isinstance(parts, Sequence):
        raise TypeError("parts must be a sequence of ScenePart")
    checked = tuple(validated_part(part) for part in parts)
    if engine is not None and engine not in (*GL_ENGINES, FALLBACK_ENGINE):
        raise ValueError(f"unknown engine {engine!r}")
    settings.view(view)
    skin = skin_vertices_of(checked)
    frame_basis = coerce_basis(basis, skin, side=side, settings=settings)
    focus = focus_vertices_of(checked)
    rig = camera_rig(focus, view, frame_basis, settings)
    require(rig.distance > 0.0, "the camera sits off the target")

    image: NDArray[np.uint8] | None = None
    mask: NDArray[np.uint8] | None = None
    used = FALLBACK_ENGINE
    if engine != FALLBACK_ENGINE:
        try:
            image, mask, used = render_mujoco(checked, rig, settings)
        except GLUnavailableError as exc:
            if engine is not None or FALLBACK_ENGINE not in settings.engine_order:
                raise
            logger.warning("falling back to matplotlib: %s", exc)
        if engine is not None and image is not None and used != engine:
            raise GLUnavailableError(f"engine {engine!r} unavailable; {used!r} works")
    if image is None:
        image = render_matplotlib(checked, rig, settings)
        used = FALLBACK_ENGINE
    if mask is None:
        fill = projected_fill(rig, focus)
    else:
        ndc = rig.project(focus)
        window = None if len(focus) == len(skin) else (ndc.min(axis=0), ndc.max(axis=0))
        fill = _measured_fill(mask, window)
    rendered = RenderedView(
        image=image,
        engine=used,
        fill_fraction=fill,
        view=view,
        camera=rig,
        kind_mask=mask,
        settings_hash=settings.settings_hash,
    )
    ensure(
        image.dtype == np.uint8
        and image.shape == (settings.frame_px, settings.frame_px, 3),
        "the frame is uint8 RGB of frame_px square",
    )
    return rendered


def render_view(
    result: Any,
    view: str,
    *,
    shaft_roll_deg: float = 0.0,
    side: Any = "lead",
    settings: HandRenderSettings | None = None,
    grip_radius_m: float | None = None,
    basis: Any = None,
    engine: str | None = None,
) -> RenderedView:
    """Render one backend result on the club from one named view.

    The club (grip rubber and a shaft stub) is added along ``+z``. The skin
    is rolled by ``shaft_roll_deg`` about the shaft; the camera basis is the
    caller's ``basis``, else one the result declares (``.render_basis``, or
    ``.components`` tagged ``dorsum`` and ``thumb``), else the heuristic on
    the unrolled skin. The camera does not follow the roll. The grip radius is ``grip_radius_m``, else
    ``result.grip_radius_m``, else the settings value.

    Raises ``ValueError`` / ``TypeError`` for an unknown view or side, an
    empty mesh, or a roll that is not finite or is outside the roll range.
    """
    settings = resolve_settings(settings)
    side_name(side)
    settings.view(view)
    unrolled = assemble_scene(
        result, shaft_roll_deg=0.0, settings=settings, grip_radius_m=grip_radius_m
    )
    if basis is None:
        basis = result_basis(result, settings=settings)
    frame_basis = coerce_basis(
        basis, skin_vertices_of(unrolled), side=side, settings=settings
    )
    parts = assemble_scene(
        result,
        shaft_roll_deg=shaft_roll_deg,
        settings=settings,
        grip_radius_m=grip_radius_m,
    )
    return render_scene(
        parts, view, basis=frame_basis, settings=settings, side=side, engine=engine
    )


def save_png(rendered: Any, path: str | Path) -> Path:
    """Write ``rendered.image`` as a PNG. The bytes are deterministic."""
    from PIL import Image

    image = np.asarray(getattr(rendered, "image", rendered))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected a uint8 RGB image, got {image.dtype} {image.shape}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(target, format="PNG", optimize=True)
    return target


def register_render_source(
    name: str, builder: Callable[..., Any], *, replace: bool = False
) -> None:
    """Register a hand builder for the contact sheet.

    ``builder(side=<"lead"|"trail">, grip_radius_m=<float>)`` returns any
    result :func:`render_view` accepts. Wrap a backend's own
    ``build_hand_visual`` in a lambda to adapt its signature.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"source name must be a non-empty string, got {name!r}")
    if not callable(builder):
        raise TypeError(f"builder for {name!r} must be callable")
    if name in _RENDER_SOURCES and not replace:
        raise ValueError(f"render source {name!r} is already registered")
    _RENDER_SOURCES[name] = builder


def render_sources() -> tuple[str, ...]:
    """Registered source names, sorted."""
    return tuple(sorted(_RENDER_SOURCES))


def build_render_source(name: str, side: Any, *, grip_radius_m: float) -> Any:
    """Call the registered builder ``name`` for one hand."""
    if name not in _RENDER_SOURCES:
        raise ValueError(f"unknown render source {name!r}; known: {render_sources()}")
    return _RENDER_SOURCES[name](side=side_name(side), grip_radius_m=grip_radius_m)


def _capsule_source(*, side: str, grip_radius_m: float) -> Any:
    """The in-repo capsule hand (visual only)."""
    return build_grip_hand_mesh(Hand(side), grip_radius_m=grip_radius_m)


def _downscale(image: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    from PIL import Image

    resized = Image.fromarray(image, mode="RGB").resize(
        (size, size), Image.Resampling.BOX
    )
    return np.asarray(resized, dtype=np.uint8)


def _label_strip(
    text: str, width: int, settings: HandRenderSettings
) -> NDArray[np.uint8]:
    from PIL import Image, ImageDraw, ImageFont

    sheet = settings.sheet
    background = tuple(int(round(c * 255)) for c in settings.background_rgba[:3])
    strip = Image.new("RGB", (width, sheet.label_height_px), background)
    try:
        font = ImageFont.load_default(size=sheet.label_font_px)
    except TypeError:  # Pillow without a scalable default font
        font = ImageFont.load_default()
    ImageDraw.Draw(strip).text(
        (sheet.margin_px, sheet.label_height_px // 2),
        text,
        fill=sheet.label_rgb,
        font=font,
        anchor="lm",
    )
    return np.asarray(strip, dtype=np.uint8)


def render_contact_sheet(
    backend_ids: Sequence[str] | None = None,
    rolls_deg: Sequence[float] | None = None,
    view: str = "palm",
    side: Any = "lead",
    settings: HandRenderSettings | None = None,
    *,
    engine: str | None = None,
) -> ContactSheet:
    """A grid of one view: rows are registered sources, columns are rolls.

    ``backend_ids`` defaults to every registered source and ``rolls_deg`` to
    ``settings.rolls_deg``. Each cell is labelled with its source and roll.
    Raises ``ValueError`` for an unknown source, view or side.
    """
    settings = resolve_settings(settings)
    names = render_sources() if backend_ids is None else tuple(backend_ids)
    rolls = settings.rolls_deg if rolls_deg is None else tuple(rolls_deg)
    if not names or not rolls:
        raise ValueError("the contact sheet needs at least one source and one roll")
    side = side_name(side)
    sheet = settings.sheet
    background = np.rint(np.asarray(settings.background_rgba[:3]) * 255).astype(
        np.uint8
    )
    rows_px, cells = [], []
    for name in names:
        result = build_render_source(name, side, grip_radius_m=settings.grip_radius_m)
        row = tuple(
            render_view(
                result,
                view,
                shaft_roll_deg=roll,
                side=side,
                settings=settings,
                engine=engine,
            )
            for roll in rolls
        )
        cells.append(row)
        tiles = []
        for roll, cell in zip(rolls, row, strict=True):
            label = f"{name} | {side} | {view} | roll {roll:+.0f} deg | {cell.engine}"
            tile = np.vstack(
                [
                    _label_strip(label, sheet.cell_px, settings),
                    _downscale(cell.image, sheet.cell_px),
                ]
            )
            margin = sheet.margin_px
            padded = np.tile(
                background, (tile.shape[0] + 2 * margin, tile.shape[1] + 2 * margin, 1)
            )
            padded[margin : margin + tile.shape[0], margin : margin + tile.shape[1]] = (
                tile
            )
            tiles.append(padded)
        rows_px.append(np.hstack(tiles))
    image = np.vstack(rows_px)
    engines = sorted({cell.engine for row in cells for cell in row})
    return ContactSheet(
        image=np.ascontiguousarray(image),
        engine="+".join(engines),
        cells=tuple(cells),
        row_labels=tuple(names),
        column_labels=tuple(f"{roll:+.0f}" for roll in rolls),
    )


register_render_source("capsule", _capsule_source, replace=True)
