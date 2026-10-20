"""Club meshes per spec body, for the rigid club and the chain shaft (R6, #12686).

The rigid club (``club.shaft.model = "rigid"``, the default) draws every club
part on the club (grip) body. With ``club.shaft.model = "chain"``
(``motion_matching.club_shaft_chain.apply_shaft_chain``) the shaft is cut at
the chain joints so each piece follows its own body: the grip, both grip
hands and the root shaft piece stay on the club body, link ``i`` carries the
shaft piece between joints ``i`` and ``i + 1`` and the tip link also carries
the head. Bending the chain then bends the drawn shaft and moves the drawn
head with the physical head.

Frames: every link body uses the club-body frame (``club_shaft_chain``: at
zero deflection all link frames coincide with the club frame), so the pieces
are cut from the same club-frame tube :func:`club_assembly.assembly_meshes`
draws, with the radius of that tube's linear taper at each cut. Visual only:
nothing here touches mass, inertia or joints.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.model_appearance.club_assembly import (
    SHAFT_BUTT_RADIUS_FRACTION,
    SHAFT_TIP_RADIUS_FRACTION,
    ClubAssembly,
    club_body_name,
    club_part_material,
)
from src.shared.python.model_appearance.club_head_mesh import load_club_head
from src.shared.python.model_appearance.club_shaft_mesh import tapered_tube
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hands import assembly_meshes_with_hands
from src.shared.python.model_appearance.schema import AppearanceDocument

#: ``(label, mesh in the spec body frame, material name)`` of one drawn part.
ClubPart = tuple[str, Mesh, str]


@dataclass(frozen=True)
class ChainLink:
    """One chain link body and its shaft span along club ``y`` (metres).

    ``start_y_m`` is the joint at the grip side, ``end_y_m`` the next joint
    (or the tip, ``0``); the shaft runs toward the grip along ``-y``.
    """

    body: str
    start_y_m: float
    end_y_m: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.start_y_m) and np.isfinite(self.end_y_m)):
            raise ValueError(f"chain link {self.body!r} span must be finite")
        if self.start_y_m >= self.end_y_m:
            raise ValueError(f"chain link {self.body!r} span must run toward the tip")


@dataclass(frozen=True)
class ShaftChainLayout:
    """The club body and its chain links in order from the grip to the tip."""

    club_body: str
    links: tuple[ChainLink, ...]

    def __post_init__(self) -> None:
        if not self.links:
            raise ValueError("a shaft chain layout needs at least one link")

    @property
    def tip_link(self) -> str:
        return self.links[-1].body

    def bodies(self) -> tuple[str, ...]:
        """Every body that carries a club part, the club body first."""
        return (self.club_body, *(link.body for link in self.links))


def chain_layout(spec: Mapping[str, Any]) -> ShaftChainLayout | None:
    """The chain layout of ``spec``, or ``None`` for a rigid shaft.

    Follows the joints from ``club.shaft.club_body`` to ``club.shaft.tip_link``
    (``apply_shaft_chain`` records both). Raises ``ValueError`` when the chain
    record and the joint tree disagree.
    """
    shaft = (spec.get("club") or {}).get("shaft") or {}
    if shaft.get("model", "rigid") != "chain":
        return None
    club_body, tip = shaft.get("club_body"), shaft.get("tip_link")
    if not club_body or not tip:
        raise ValueError("a chain shaft record needs club_body and tip_link")
    parent_joint = {j["child"]: j for j in spec["joints"]}
    joints: list[Mapping[str, Any]] = []
    child = str(tip)
    while child != club_body:  # walk tip -> club body (one parent per body)
        joint = parent_joint.get(child)
        if joint is None or len(joints) >= len(spec["joints"]):
            raise ValueError(f"no joint chain from {club_body!r} to {tip!r}")
        joints.insert(0, joint)
        child = joint["parent"]
    starts = [float(np.asarray(j["parent_to_base"])[1, 3]) for j in joints]
    ends = [*starts[1:], 0.0]
    links = tuple(
        ChainLink(str(j["child"]), s, e)
        for j, s, e in zip(joints, starts, ends, strict=True)
    )
    return ShaftChainLayout(str(club_body), links)


def _shaft_span(club: ClubAssembly) -> tuple[float, float]:
    """Club-frame ``y`` of the drawn shaft: (grip end, hosel top)."""
    return -(club.length_m - club.grip_length_m), -load_club_head(
        club.head_alias
    ).hosel_top_m


def shaft_piece_meshes(club: ClubAssembly, layout: ShaftChainLayout) -> dict[str, Mesh]:
    """The drawn shaft cut at the chain joints, one tube per body.

    Postcondition: the pieces tile the rigid shaft tube of
    ``assembly_meshes(club)`` (same ends, same linear taper). Raises
    ``ValueError`` when a body's span misses the drawn shaft (e.g. a joint
    inside the hosel).
    """
    grip_y, top_y = _shaft_span(club)
    r_top = SHAFT_TIP_RADIUS_FRACTION * club.shaft_radius_m
    r_grip = SHAFT_BUTT_RADIUS_FRACTION * club.shaft_radius_m

    def radius(y: float) -> float:
        return r_grip + (r_top - r_grip) * (y - grip_y) / (top_y - grip_y)

    spans = [(layout.club_body, grip_y, layout.links[0].start_y_m)]
    spans += [(link.body, link.start_y_m, link.end_y_m) for link in layout.links]
    z = club.axis_offset_m
    out: dict[str, Mesh] = {}
    for body, low, high in spans:
        low, high = max(low, grip_y), min(high, top_y)
        if high <= low:
            raise ValueError(f"chain body {body!r} has no drawn shaft span")
        out[body] = tapered_tube(
            np.array([0.0, high, z]),
            np.array([0.0, low, z]),
            radius(high),
            radius(low),
        )
    return out


def club_meshes_by_body(
    spec: Mapping[str, Any],
    club: ClubAssembly,
    finish: str,
    doc: AppearanceDocument,
) -> dict[str, list[ClubPart]]:
    """Club parts per spec body (module docstring), in each body's spec frame.

    Rigid shaft: every part on the club body, in ``assembly_meshes_with_hands``
    order. Chain shaft: shaft pieces labelled ``shaft`` (club body) and
    ``shaft_link<i>``; the head on the tip link. Every label is unique.
    """
    parts = assembly_meshes_with_hands(club)
    layout = chain_layout(spec)
    if layout is None:
        body = club_body_name(spec)
        if body is None:
            raise ValueError("spec has no club body")
        return {
            body: [
                (label, mesh, club_part_material(label, finish, doc))
                for label, mesh in parts.items()
            ]
        }
    shaft_material = club_part_material("shaft", finish, doc)
    pieces = shaft_piece_meshes(club, layout)
    out: dict[str, list[ClubPart]] = {
        layout.club_body: [("shaft", pieces[layout.club_body], shaft_material)]
    }
    out[layout.club_body] += [
        (label, mesh, club_part_material(label, finish, doc))
        for label, mesh in parts.items()
        if label not in ("shaft", "head")
    ]
    for i, link in enumerate(layout.links):
        out[link.body] = [(f"shaft_link{i}", pieces[link.body], shaft_material)]
    out[layout.tip_link].append(
        ("head", parts["head"], club_part_material("head", finish, doc))
    )
    return out
