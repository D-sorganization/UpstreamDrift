"""Grip-size sweep keyed to hand length (HRG-GD4 PR 2, #13892).

Golf Pride bins grip size by hand length from the wrist crease to the middle
fingertip: undersize 5-6.5 in, standard 6.6-7.5 in, midsize 7.6-9 in, jumbo
9.1-10 in (https://www.golfpride.com/us/en-us/grip-academy/swing-grip-size-guide.html).
UD's ``hand_length_m`` is ANSUR II ``handlength`` (#13760), measured from the
stylion (ISO 7250-1 4.3.1), which lies proximal of the crease, so the two are
not equivalent. AIST's hand data measure both on the same 530 hands (L01
stylion 184.2 mm, L02 crease 178.0 mm pooled means;
https://www.airc.aist.go.jp/dhrt/hand/data/L.pdf), giving
:data:`STYLION_TO_CREASE_M`. Bins report the corrected and uncorrected bin
and flag lengths within :data:`BIN_EDGE_BAND_IN` of an edge with both
neighbours. ``GripGeometry.cap_offset_m`` is not applied: stations are
measured from the butt, as in ``GripSpec``. Rows are in SI units.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from src.shared.python.core.handedness import lead_side, trail_side
from src.shared.python.grip_contact.contact_topology import (
    HAND_ROLES,
    ContactTopology,
    contact_topology,
    site_grip_surface_gap_m,
)
from src.shared.python.grip_contact.friction_band import governing_branch
from src.shared.python.grip_contact.grip_geometry import get_grip_preset
from src.shared.python.grip_contact.pad_layout import (
    peak_bushing_demand,
    required_squeeze_n,
)
from src.shared.python.hand_size import validate_hand_length_m
from src.shared.python.motion_matching.grip_contract import GripSpec

INCH_M: Final[float] = 0.0254
STYLION_TO_CREASE_M: Final[float] = 0.0062  # AIST L01 - L02, pooled means
GOLF_PRIDE_BIN_UPPER_EDGES_IN: Final[tuple[tuple[str, float], ...]] = (
    ("undersize", 6.55),
    ("standard", 7.55),
    ("midsize", 9.05),
    ("jumbo", math.inf),
)
BIN_EDGE_BAND_IN: Final[float] = 0.05
# Jumbo has no sourced GripGeometry preset (#13891), so it maps to None.
BIN_PRESETS: Final[dict[str, str | None]] = {
    "undersize": "undersize",
    "standard": "mens_standard",
    "midsize": "midsize",
    "jumbo": None,
}

HandDemand = tuple[float, float, float]


@dataclass(frozen=True)
class GripSizeBin:
    """Golf Pride bin of one hand length; ``edge_neighbours`` is ``(below,
    above)`` within :data:`BIN_EDGE_BAND_IN` of an edge, else ``None``."""

    hand_length_m: float
    crease_length_in: float
    bin: str
    uncorrected_bin: str
    preset: str | None
    edge_neighbours: tuple[str, str] | None


def _bin_of(length_in: float) -> str:
    return next(n for n, edge in GOLF_PRIDE_BIN_UPPER_EDGES_IN if length_in < edge)


def golf_pride_bin(hand_length_m: float) -> GripSizeBin:
    """Bin of an ANSUR ``hand_length_m`` after the stylion-to-crease offset.

    Raises:
        ValueError: when ``hand_length_m`` is outside #13760's bounds.
    """
    hand_length_m = validate_hand_length_m(hand_length_m)
    crease_in = (hand_length_m - STYLION_TO_CREASE_M) / INCH_M
    edges = GOLF_PRIDE_BIN_UPPER_EDGES_IN
    neighbours = next(
        (
            (below, above)
            for (below, edge), (above, _) in zip(edges, edges[1:], strict=False)
            if abs(crease_in - edge) <= BIN_EDGE_BAND_IN
        ),
        None,
    )
    size, uncorrected = _bin_of(crease_in), _bin_of(hand_length_m / INCH_M)
    return GripSizeBin(
        hand_length_m, crease_in, size, uncorrected, BIN_PRESETS[size], neighbours
    )


def per_hand_bushing_demand(
    force_on_club_n: Mapping[str, np.ndarray],
    torque_on_club_nm: Mapping[str, np.ndarray],
    club_rotation: np.ndarray,
    grip_axis_club: np.ndarray,
    handedness: str = "right",
) -> dict[str, HandDemand]:
    """:func:`peak_bushing_demand` of each hand role; ``L``/``R`` map to lead
    and trail through ``handedness``."""
    sides = {"lead": lead_side(handedness), "trail": trail_side(handedness)}
    return {
        hand: peak_bushing_demand(
            {side: force_on_club_n[side]},
            {side: torque_on_club_nm[side]},
            club_rotation,
            grip_axis_club,
        )
        for hand, side in sides.items()
    }


@dataclass(frozen=True)
class SweepRow:
    """One preset x build-up x hand-length case; per-hand fields by role."""

    hand_length_m: float
    preset: str
    buildup_layers: int
    size_bin: GripSizeBin
    radius_m: dict[str, float]
    squeeze_n: dict[str, float]
    torque_branch_n: dict[str, float]
    governing: dict[str, str]
    pad_normal_deg: dict[str, tuple[float, ...]]


def _pad_normal_deg(topology: ContactTopology, hand: str) -> tuple[float, ...]:
    """Sorted angles about the grip axis of ``hand``'s sites touching the grip."""
    geom = topology.grip_geometry
    touching = [
        s
        for s in topology.sites
        if s.hand == hand and site_grip_surface_gap_m(s, grip_geometry=geom) <= 0.0
    ]
    angles = (math.degrees(math.atan2(s.centre_m[2], s.centre_m[1])) for s in touching)
    return tuple(sorted(round(a, 1) + 0.0 for a in angles))


def grip_size_sweep(
    hand_lengths_m: Sequence[float],
    presets: Sequence[str],
    buildup_layers: Sequence[int],
    demand: Mapping[str, HandDemand],
    friction: float,
    spec: GripSpec | None = None,
) -> tuple[SweepRow, ...]:
    """Rows ordered hand length, preset, layers. Each hand's squeeze is
    :func:`required_squeeze_n` of its own demand at its own station's radius;
    ``torque_branch_n`` is ``tau / (mu r)`` and ``governing`` D1's branch.

    Raises:
        ValueError: on a demand without both hand roles, an unknown preset,
            a hand length outside #13760's bounds or a pad outside the grip.
    """
    if set(demand) != set(HAND_ROLES):
        raise ValueError("demand must be keyed by lead and trail")
    rows = []
    for hand_length_m in hand_lengths_m:
        size_bin = golf_pride_bin(hand_length_m)
        for preset in presets:
            for layers in buildup_layers:
                geometry = get_grip_preset(preset).with_buildup(layers)
                topo = contact_topology(spec, hand_length_m, grip_geometry=geometry)
                radius = {h: topo.hand_grip_radius_m(h) for h in HAND_ROLES}
                args = {h: (*demand[h], friction, radius[h]) for h in HAND_ROLES}
                rows.append(
                    SweepRow(
                        hand_length_m,
                        preset,
                        layers,
                        size_bin,
                        radius,
                        {h: required_squeeze_n(*args[h]) for h in HAND_ROLES},
                        {h: demand[h][1] / (friction * radius[h]) for h in HAND_ROLES},
                        {h: governing_branch(*args[h]) for h in HAND_ROLES},
                        {h: _pad_normal_deg(topo, h) for h in HAND_ROLES},
                    )
                )
    return tuple(rows)
