"""ZTCF/ZVCF hand-on-club arrows for the native export (#15685 K5, #15150 Z3b).

Per frame, each hand's load on the club splits into its drift part (ZTCF,
``tau = 0``: velocity terms, gravity and passive springs) and its control
part (ZVCF, the applied torques alone, no gravity, so gravity is counted
once). One render shows one view of that split (:data:`PART_VIEWS`: tip to
tail with the total, the ZTCF alone, the ZVCF alone, or the total) of the
force or the torque, for the lead hand, the trail hand or both, each drawn
from its own point. The split is evaluated pointwise on the closed-loop
Pinocchio provider of the bundle's own specification
(``hand_wrench_split``), so it needs a dynamics bundle whose efforts are the
torques that produced the motion; a kinematic replay (zero efforts) is
refused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import ArrowGlyph, ForceGlyphStyle
from src.shared.python.force_overlay.tip_to_tail import (
    TipToTailStyle,
    colour_key,
    part_arrows,
)
from src.shared.python.simulation_backends.hand_force_split import HANDS

__all__ = [
    "HAND_CHOICES",
    "HAND_COLOURS_PATH",
    "LEGEND",
    "gained_style",
    "hand_arrows",
    "hand_styles",
    "hands_colour_key",
    "hands_of",
    "legend_for",
    "ztcf_zvcf_arrows",
]

#: ``--ztcf-zvcf-hands``: both hands or one.
HAND_CHOICES = ("both", *HANDS)
_HAND_LABEL = {"lead": "Lead", "trail": "Trail"}
_VIEW_TEXT = {
    "tip_to_tail": "ZTCF drift + ZVCF control = total, tip to tail",
    "ztcf": "ZTCF drift (passive, velocity and gravity) alone",
    "zvcf": "ZVCF control (applied torques alone) alone",
    "total": "total (ZTCF + ZVCF)",
}
_NULL_SPACE = "one null-space solution"
HAND_COLOURS_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "ztcf_zvcf_hand_colours.json"
)
_COLOUR_FIELDS = ("drift_hex", "control_hex", "total_hex")


def hand_styles(path: Path | None = None) -> dict[str, TipToTailStyle]:
    """Each hand's arrow colours from the settings file.

    Raises:
        ValueError: a hand or one of its colours is missing, or a colour is
            not ``#RRGGBB``.
    """
    document = json.loads((path or HAND_COLOURS_PATH).read_text(encoding="utf-8"))
    styles = {}
    for hand in HANDS:
        entry = document.get(hand)
        if not isinstance(entry, dict) or any(f not in entry for f in _COLOUR_FIELDS):
            raise ValueError(f"hand colours need {_COLOUR_FIELDS} for {hand!r}")
        styles[hand] = TipToTailStyle(**{f: entry[f] for f in _COLOUR_FIELDS})
    return styles


def hands_colour_key(
    view: str,
    hands: Sequence[str],
    styles: Mapping[str, TipToTailStyle] | None = None,
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    """The colour key of ``view`` for each of ``hands``, named by hand."""
    looks = styles or hand_styles()
    return tuple(
        (f"{_HAND_LABEL[hand]} {text}", rgb)
        for hand in hands
        for text, rgb in colour_key(view, looks[hand])
    )


def hands_of(choice: str) -> tuple[str, ...]:
    """The hands a ``--ztcf-zvcf-hands`` choice draws; ``ValueError`` otherwise."""
    if choice not in HAND_CHOICES:
        raise ValueError(f"hands must be one of {HAND_CHOICES}, got {choice!r}")
    return HANDS if choice == "both" else (choice,)


def _hands_text(hands: Sequence[str]) -> str:
    names = [_HAND_LABEL[h].lower() for h in hands]
    return " and ".join(names).capitalize()


def _legend(view: str, quantity: str, hands: Sequence[str]) -> str:
    return (
        f"{_hands_text(hands)} hand {quantity} on club: {_VIEW_TEXT[view]}; "
        f"{_NULL_SPACE}"
    )


LEGEND = _legend("tip_to_tail", "force", HANDS)


def _checked_gain(gain: float) -> float:
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError(
            f"the hand-force arrow gain must be finite and positive, got {gain}"
        )
    return float(gain)


def gained_style(style: ForceGlyphStyle, gain: float) -> ForceGlyphStyle:
    """``style`` with the hand-force (grip) arrows ``gain`` times longer.

    Trail-hand forces are tens of newtons, a few centimetres at body-weight
    scale; the gain makes them legible without touching any other arrow.
    Returns ``style`` itself for a gain of one.

    Raises:
        ValueError: ``gain`` is not finite and positive.
    """
    if _checked_gain(gain) == 1.0:
        return style
    grip = WrenchKind.GRIP
    kind_scale = {**style.kind_scale, grip: style.kind_scale.get(grip, 1.0) * gain}
    return replace(style, kind_scale=kind_scale)


def legend_for(
    gain: float,
    view: str = "tip_to_tail",
    quantity: str = "force",
    hands: Sequence[str] = HANDS,
) -> str:
    """The legend, naming hands, view, quantity and any arrow gain."""
    text = _legend(view, quantity, hands)
    if _checked_gain(gain) == 1.0:
        return text
    return f"{text}; hand arrows drawn {gain:g}x"


def hand_arrows(
    split: Any,
    style: ForceGlyphStyle,
    view: str,
    quantity: str,
    hands: Sequence[str],
    colours: TipToTailStyle | None = None,
    gain: float = 1.0,
    row: int = 0,
) -> tuple[ArrowGlyph, ...]:
    """Arrows of ``view`` for each of ``hands`` at ``row`` of a hand-wrench split.

    Each hand is drawn from its own point, labelled with its name and in its
    own colours (:func:`hand_styles`; ``colours`` paints every hand alike).
    The length limit grows with ``gain``, so lengthening the arrows never
    turns a legible frame into an error.
    """
    reach = replace(style, max_length_m=style.max_length_m * _checked_gain(gain))
    looks = hand_styles() if colours is None else dict.fromkeys(hands, colours)
    field = "force_n" if quantity == "force" else "torque_nm"
    out: list[ArrowGlyph] = []
    for hand in hands:
        drift, control = split.part(hand, "ztcf"), split.part(hand, "zvcf")
        out.extend(
            part_arrows(
                drift.point_m[row],
                getattr(drift, field)[row],
                getattr(control, field)[row],
                reach,
                view,
                quantity,
                looks[hand],
                _HAND_LABEL[hand],
            )
        )
    return tuple(out)


def _closure_provider(
    spec_bytes: bytes, order: tuple[str, ...]
) -> tuple[Any, Any, Any, Any]:
    """``(provider, grip frame, club load, lead point)`` in ``order``."""
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.loop_nullspace.provider import (
        AdapterConstrainedProvider,
        ReorderedProvider,
    )

    adapter = PinocchioForceAdapter(spec_bytes)
    native = tuple(adapter.coordinate_order)
    provider: Any = AdapterConstrainedProvider(adapter)
    if native == order:
        return (
            provider,
            adapter.closure_wrench_frame,
            adapter.club_subtree_load,
            adapter.lead_wrist_point,
        )
    take = np.array([order.index(n) for n in native])

    def native_of(x: Any) -> np.ndarray:
        return np.asarray(x)[take]

    return (
        ReorderedProvider(provider, native, order),
        lambda q: adapter.closure_wrench_frame(native_of(q)),
        lambda q, v, a, gravity: adapter.club_subtree_load(
            native_of(q), native_of(v), native_of(a), gravity
        ),
        lambda q: adapter.lead_wrist_point(native_of(q)),
    )


def _states(swing: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(q, v, tau)`` per source state of ``swing`` (efforts held per step)."""
    bundle = swing.bundle
    q = np.asarray(swing.q, dtype=float)
    times = np.arange(q.shape[0], dtype=float) * float(bundle.dt_s)
    if swing.v is not None:
        v = np.asarray(swing.v, dtype=float)
    elif q.shape == bundle.reference_v.shape:
        v = np.asarray(bundle.reference_v, dtype=float)
    else:
        v = np.gradient(q, times, axis=0)
    efforts = np.asarray(bundle.efforts, dtype=float)
    step = np.clip((times / float(bundle.dt_s)).astype(int), 0, efforts.shape[0] - 1)
    return q, v, efforts[step]


def ztcf_zvcf_arrows(
    swing: Any,
    style: ForceGlyphStyle,
    colours: TipToTailStyle | None = None,
    view: str = "tip_to_tail",
    quantity: str = "force",
    hands: Sequence[str] = HANDS,
    gain: float = 1.0,
) -> Callable[[int], tuple[ArrowGlyph, ...]]:
    """Arrows by source state index, evaluated lazily and cached.

    Raises:
        ValueError: the bundle has no efforts (a kinematic replay), or its
            specification declares no closure.
    """
    from src.shared.python.simulation_backends.hand_force_split import (
        hand_wrench_split,
    )

    bundle = swing.bundle
    if not np.any(np.asarray(bundle.efforts)):
        raise ValueError(
            "the ZTCF/ZVCF hand force needs a dynamics bundle with efforts; this "
            "is a kinematic replay (run the kinetic match first)"
        )
    if "closure" not in json.loads(bundle.spec_bytes):
        raise ValueError("the specification declares no hand-club closure")
    provider, frame_at, club_load, lead_point = _closure_provider(
        bundle.spec_bytes, tuple(bundle.coordinate_order)
    )
    q, v, tau = _states(swing)
    cache: dict[int, tuple[ArrowGlyph, ...]] = {}

    def arrows(index: int) -> tuple[ArrowGlyph, ...]:
        k = min(max(int(index), 0), q.shape[0] - 1)
        if k not in cache:
            split = hand_wrench_split(
                provider, frame_at, club_load, lead_point, q[k], v[k], tau[k]
            )
            cache[k] = hand_arrows(split, style, view, quantity, hands, colours, gain)
        return cache[k]

    return arrows
