"""Shaft internal loads drawn in the native export (#15150 Track Z, Z9c).

One arrow per chain-shaft joint: the bending moment (right-hand vector) or the
shear the proximal side applies to the tip-ward chain, world frame, normal to
the shaft (``shaft_load_split``), of the total or one ZTCF/ZVCF part, evaluated
on the hand split's grounded provider (:func:`ztcf_feed.split_provider`), so
the parts add to the total. The look is ``shaft_loads_settings.json``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import ArrowGlyph
from src.shared.python.force_overlay.palette import hex_to_rgba
from src.shared.python.simulation_backends.shaft_load_split import (
    ShaftLoadSplit,
    ShaftSectionLoads,
)

QUANTITIES = ("bending", "shear")
#: the first part is the default (the total, ZTCF + ZVCF)
PARTS = ("total", "ztcf", "zvcf")
SETTINGS_PATH = Path(__file__).with_name("shaft_loads_settings.json")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
_QUANTITY_KEYS = {"units", "scale_m", "floor", "legend", "key"}


@dataclass(frozen=True)
class ShaftLoadLook:
    """Arrow look: ``scale_m`` arrow metres per unit, ``floor`` smallest load drawn."""

    max_length_m: float
    shaft_radius_m: float
    head_radius_ratio: float
    head_length_ratio: float
    station_stride: int
    quantities: Mapping[str, Mapping[str, Any]]
    parts: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if set(self.quantities) != set(QUANTITIES) or set(self.parts) != set(PARTS):
            raise ValueError(f"settings need quantities {QUANTITIES} and parts {PARTS}")
        if any(set(q) != _QUANTITY_KEYS for q in self.quantities.values()):
            raise ValueError(f"each quantity needs exactly {sorted(_QUANTITY_KEYS)}")
        sizes = (self.max_length_m, self.shaft_radius_m, self.head_radius_ratio)
        scales = tuple(q["scale_m"] for q in self.quantities.values())
        if not all(float(x) > 0.0 for x in (*sizes, *scales, self.head_length_ratio)):
            raise ValueError("scale_m, lengths, radii and ratios must be positive")
        if any(float(q["floor"]) < 0.0 for q in self.quantities.values()):
            raise ValueError("floor must not be negative")
        if not isinstance(self.station_stride, int) or self.station_stride < 1:
            raise ValueError("station_stride must be a positive integer")
        if not all(_HEX.match(p.get("hex", "")) for p in self.parts.values()):
            raise ValueError("part colours must be #RRGGBB")


def load_look(path: Path | None = None) -> ShaftLoadLook:
    """Read :class:`ShaftLoadLook` (``ValueError`` on an invalid value)."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    for key in ("schema_version", "description", "sources"):
        raw.pop(key, None)
    return ShaftLoadLook(**raw)


def validate(quantity: str | None, part: str, overlays: bool = True) -> None:
    """``ValueError`` for an unknown quantity or part, or loads without overlays."""
    if part not in PARTS:
        raise ValueError(f"shaft-load part must be one of {PARTS}, got {part!r}")
    if quantity is not None and quantity not in QUANTITIES:
        raise ValueError(f"shaft-load quantity must be one of {QUANTITIES}")
    if quantity is not None and not overlays:
        raise ValueError("shaft loads need overlays (omit --no-overlay)")


def _arrow(
    tail: np.ndarray, load: np.ndarray, spec: Mapping[str, Any], look: ShaftLoadLook
) -> dict[str, Any]:
    """``ArrowGlyph`` fields of ``load`` at ``tail``, clamped to ``max_length_m``."""
    vector = float(spec["scale_m"]) * load
    length = float(np.linalg.norm(vector))
    clamped = length > look.max_length_m
    if clamped:
        vector = vector * (look.max_length_m / length)
    tip, head = tail + vector, tail + (1.0 - look.head_length_ratio) * vector
    return {
        "kind": WrenchKind.JOINT_REACTION,
        "tail_m": tuple(map(float, tail)),
        "tip_m": tuple(map(float, tip)),
        "head_base_m": tuple(map(float, head)),
        "shaft_radius_m": look.shaft_radius_m,
        "head_radius_m": look.shaft_radius_m * look.head_radius_ratio,
        "magnitude": float(np.linalg.norm(load)),
        "units": spec["units"],
        "clamped": clamped,
    }


def shaft_load_arrows(
    points: np.ndarray,
    loads: ShaftSectionLoads,
    quantity: str,
    part: str,
    look: ShaftLoadLook,
    row: int = 0,
) -> tuple[ArrowGlyph, ...]:
    """One arrow per drawn shaft joint (every ``station_stride``-th, grip first).

    Each is the load times ``scale_m`` from its joint, clamped to
    ``max_length_m`` (``clamped`` set) and left out below ``floor``;
    ``magnitude`` is the load. ``ValueError`` unless ``points`` match ``loads``.
    """
    validate(quantity, part)
    pts = np.asarray(points, dtype=float)
    field = loads.bending_nm if quantity == "bending" else loads.shear_n
    vectors = np.asarray(field[row], dtype=float)
    if pts.shape != vectors.shape:
        raise ValueError(f"points {pts.shape} do not match the loads {vectors.shape}")
    spec, rgba = look.quantities[quantity], hex_to_rgba(look.parts[part]["hex"])
    return tuple(
        ArrowGlyph(
            label=f"shaft_{quantity}:{part}:{i}",
            rgba=rgba,
            **_arrow(pts[i], vectors[i], spec, look),
        )
        for i in range(0, pts.shape[0], look.station_stride)
        if float(np.linalg.norm(vectors[i])) >= float(spec["floor"])
    )


def legend(quantity: str, part: str, look: ShaftLoadLook) -> str:
    """Legend naming the quantity, its units, the sign convention and scale."""
    spec = look.quantities[quantity]
    per_m = 1.0 / float(spec["scale_m"])
    return str(spec["legend"]).format(part=look.parts[part]["text"], per_m=per_m)


def colour_key(
    quantity: str, part: str, look: ShaftLoadLook
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    """The one ``(text, 8-bit RGB)`` colour-key row of the drawn part."""
    r, g, b, _ = hex_to_rgba(look.parts[part]["hex"])
    text = str(look.quantities[quantity]["key"]).format(part=look.parts[part]["text"])
    return ((text, (round(255 * r), round(255 * g), round(255 * b))),)


def with_shaft_loads(
    feed: Any,
    glyphs: Callable[[int], tuple[Any, ...]],
    quantity: str,
    part: str,
    look: ShaftLoadLook,
) -> Any:
    """``feed`` with the shaft arrows after its own extras, legend and key."""
    prior = feed.extra_arrows
    text = legend(quantity, part, look)
    return replace(
        feed,
        extra_arrows=glyphs if prior is None else (lambda i: (*prior(i), *glyphs(i))),
        extra_legend=f"{feed.extra_legend} | {text}" if feed.extra_legend else text,
        colour_key=(*feed.colour_key, *colour_key(quantity, part, look)),
    )


def _sampler(swing: Any) -> Callable[[int], tuple[ShaftLoadSplit, np.ndarray]]:
    """``k -> (shaft-load split, joint points)`` at source state ``k``."""
    from src.shared.python.simulation_backends.shaft_load_split import (
        shaft_load_split,
    )
    from src.tools.native_viewer_export.ztcf_feed import _states, split_provider

    bundle = swing.bundle
    if not np.any(np.asarray(bundle.efforts)):
        raise ValueError("shaft loads need a dynamics bundle, not a kinematic replay")
    split = split_provider(bundle.spec_bytes, tuple(bundle.coordinate_order))
    stations, link_loads = split.stations, split.link_loads
    if stations is None or link_loads is None:
        raise ValueError("the split provider exposes no shaft stations")
    q, v, tau = _states(swing)

    def sample(index: int) -> tuple[ShaftLoadSplit, np.ndarray]:
        k = min(max(int(index), 0), q.shape[0] - 1)
        q1, v1, t1 = q[k : k + 1], v[k : k + 1], tau[k : k + 1]
        loads = shaft_load_split(split.provider, stations, link_loads, q1, v1, t1)
        return loads, stations(q[k])[0]

    return sample


def feed_with_options(feed: Any, swing: Any, opts: Any) -> Any:
    """``feed`` plus the shaft arrows ``opts`` asks for; unchanged when off."""
    if (quantity := opts.shaft_loads) is None:
        return feed
    part, look = opts.shaft_loads_part, load_look()
    glyphs = shaft_load_glyphs(swing, quantity, part, look)
    return with_shaft_loads(feed, glyphs, quantity, part, look)


def add_cli_arguments(parser: Any) -> None:
    """Add ``--shaft-loads`` and ``--shaft-loads-part`` to an argparse parser."""
    parser.add_argument(
        "--shaft-loads",
        choices=QUANTITIES,
        help="draw the shaft's bending or shear at each joint (default: off)",
    )
    parser.add_argument(
        "--shaft-loads-part",
        choices=PARTS,
        default=PARTS[0],
        help="shaft-load part: total (default), ZTCF or ZVCF",
    )


def shaft_load_glyphs(
    swing: Any, quantity: str, part: str, look: ShaftLoadLook | None = None
) -> Callable[[int], tuple[ArrowGlyph, ...]]:
    """Shaft arrows by source state, lazy and cached; ``ValueError`` for a
    kinematic replay, and on first use for a rigid club (no shaft joints)."""
    validate(quantity, part)
    style = look or load_look()
    sample = _sampler(swing)
    cache: dict[int, tuple[ArrowGlyph, ...]] = {}

    def glyphs(index: int) -> tuple[ArrowGlyph, ...]:
        if index not in cache:
            split, points = sample(index)
            loads = split.total() if part == "total" else split.part(part)
            cache[index] = shaft_load_arrows(points, loads, quantity, part, style)
        return cache[index]

    return glyphs
