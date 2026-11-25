"""Midhands equivalent force and couple of both hands on the club (#15720).

Both hands' loads reduce to one force at the midpoint of the hands plus a
couple, drawn per case in its colour (``midhands_settings.json``): ZTCF is
the drift (passive, velocity and gravity), ZVCF the applied torques alone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.biomechanics.grip_wrench import HandWrench as GripHand
from src.shared.python.biomechanics.grip_wrench import analyze_grip
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    TorqueArcGlyph,
    build_glyphs,
)
from src.shared.python.force_overlay.palette import hex_to_rgba

__all__ = [
    "MIDHANDS_CASES",
    "MIDHANDS_SETTINGS_PATH",
    "MidhandsSettings",
    "case_wrench",
    "equivalent_wrench",
    "load_midhands_settings",
    "midhands_glyphs",
    "midhands_key",
]

MIDHANDS_SETTINGS_PATH = Path(__file__).with_name("midhands_settings.json")
MIDHANDS_CASES = ("ztcf", "zvcf", "total")
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")
Vec = NDArray[np.float64]
#: the shared grip labels of the midpoint force and couple (``grip_wrench``)
_NET_LABEL, _COUPLE_LABEL = "grip:net_midpoint", "grip:couple_midpoint"


def _t(v: Vec) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


@dataclass(frozen=True)
class MidhandsSettings:
    """Per-case colour and wording of the midhands glyphs, key and legend."""

    case_hex: dict[str, str]
    case_text: dict[str, str]
    key_text: str
    gain_text: str

    def rgba(self, case: str) -> tuple[float, float, float, float]:
        """RGBA of ``case``; ``ValueError`` for an unknown case."""
        return hex_to_rgba(self.case_hex[_checked_case(case)])

    def legend(self, case: str, gain: float) -> str:
        """The key text of ``case``, naming the arrow gain when it is not 1."""
        text = f"{self.case_text[_checked_case(case)]}: {self.key_text}"
        return text if gain == 1.0 else f"{text}; {self.gain_text.format(gain=gain)}"


def _checked_case(case: str) -> str:
    if case not in MIDHANDS_CASES:
        raise ValueError(f"case must be one of {MIDHANDS_CASES}, got {case!r}")
    return case


def load_midhands_settings(path: Path | None = None) -> MidhandsSettings:
    """The midhands settings; ``ValueError`` on a missing or malformed entry."""
    doc = json.loads((path or MIDHANDS_SETTINGS_PATH).read_text(encoding="utf-8"))
    for key in ("case_hex", "case_text"):
        entry = doc.get(key)
        if not isinstance(entry, dict) or set(entry) != set(MIDHANDS_CASES):
            raise ValueError(f"midhands {key} needs exactly {MIDHANDS_CASES}")
    for case, hex_code in doc["case_hex"].items():
        if not isinstance(hex_code, str) or not _HEX.match(hex_code):
            raise ValueError(f"midhands case_hex of {case!r} must be #RRGGBB")
    for key in ("key_text", "gain_text"):
        if not isinstance(doc.get(key), str) or not doc[key]:
            raise ValueError(f"midhands {key} must be a non-empty string")
    return MidhandsSettings(
        dict(doc["case_hex"]), dict(doc["case_text"]), doc["key_text"], doc["gain_text"]
    )


def equivalent_wrench(points: Any, forces: Any, torques: Any) -> tuple[Vec, Vec, Vec]:
    """``(midpoint, force, couple)`` of the lead and trail rows ``(2, 3)``.

    Via the shared :func:`analyze_grip`: ``F = F_lead + F_trail`` and
    ``M = sum(m_i + (p_i - p_mid) x F_i)`` about ``p_mid``, the midpoint, so
    the pair has the hand wrenches' moment about every point.

    Raises:
        ValueError: an array is not shape ``(2, 3)`` or not finite.
    """
    arrays = [np.asarray(a, dtype=float) for a in (points, forces, torques)]
    if any(a.shape != (2, 3) for a in arrays):
        raise ValueError(
            f"two hand wrenches need shape (2, 3), got {[a.shape for a in arrays]}"
        )
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("hand wrenches must be finite")
    p, f, t = arrays
    hands = [GripHand(side, p[i], f[i], t[i]) for i, side in enumerate("LR")]
    grip = analyze_grip(*hands, split_method="loop_nullspace_family")
    return (
        np.asarray(grip.midpoint_m),
        np.asarray(grip.net_force_n),
        np.asarray(grip.couple_at_midpoint_nm),
    )


def case_wrench(split: Any, case: str, row: int = 0) -> tuple[Vec, Vec, Vec]:
    """``(midpoint, force, couple)`` of ``case`` (a part, or ``total`` their
    sum) of a hand-wrench split; ``ValueError`` on an unknown ``case``."""
    _checked_case(case)
    hands = ("lead", "trail")
    loads = [split.total(h) if case == "total" else split.part(h, case) for h in hands]
    return equivalent_wrench(
        [w.point_m[row] for w in loads],
        [w.force_n[row] for w in loads],
        [w.torque_nm[row] for w in loads],
    )


def midhands_glyphs(
    split: Any,
    case: str,
    style: ForceGlyphStyle,
    settings: MidhandsSettings | None = None,
    row: int = 0,
) -> tuple[ArrowGlyph | TorqueArcGlyph, ...]:
    """The equivalent force arrow at the midpoint of the hands and couple arc
    about it, in the case colour, of ``case`` at ``row``; parts under the
    style's floors are left out. ``ValueError`` on an unknown ``case``."""
    look = settings or load_midhands_settings()
    mid, force, couple = case_wrench(split, case, row)
    wrenches = (
        OverlayWrench(WrenchKind.GRIP, label, "club", _t(mid), f, t, source="midhands")
        for label, f, t in (
            (_NET_LABEL, _t(force), None),
            (_COUPLE_LABEL, None, _t(couple)),
        )
    )
    painted = replace(
        style,
        kinds=frozenset({WrenchKind.GRIP}),
        groups=frozenset({"grip_net", "grip_couple"}),
        palette={WrenchKind.GRIP.value: look.case_hex[case]},
    )
    glyphs = build_glyphs(ForceTorqueFrame(0.0, "midhands", tuple(wrenches)), painted)
    return (*glyphs.arrows, *glyphs.torque_arcs)


def midhands_key(
    case: str, settings: MidhandsSettings | None = None
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    """The colour key row naming the midhands force and couple of ``case``."""
    look = settings or load_midhands_settings()
    r, g, b = (round(255 * c) for c in look.rgba(case)[:3])
    return ((look.legend(case, 1.0), (r, g, b)),)
