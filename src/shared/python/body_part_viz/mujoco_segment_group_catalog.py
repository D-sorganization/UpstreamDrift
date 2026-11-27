"""Heuristic ``SegmentGroupCatalog`` from MuJoCo body names (golf humanoid)."""

from __future__ import annotations

from collections.abc import Iterable

from .segment_angular_momentum import SegmentGroupCatalog

_LEGS = (
    "thigh",
    "shin",
    "shank",
    "foot",
    "leg",
    "ankle",
    "knee",
    "femur",
    "tibia",
    "toe",
    "talus",
    "calcan",
)
_CLUB = (
    "shaft",
    "hosel",
    "grip",
)
_TORSO = (
    "torso",
    "trunk",
    "spine",
    "chest",
    "abdomen",
    "lumbar",
    "head",
    "neck",
    "skull",
    "thorax",
    "rib",
)
_ARMS = (
    "arm",
    "hand",
    "finger",
    "thumb",
    "palm",
    "phalan",
    "wrist",
    "elbow",
    "shoulder",
    "clavicle",
    "scapula",
    "humerus",
    "radius",
    "ulna",
    "forearm",
)

# Bodies excluded from the five-group partition and whole-body COM / H reference.
GROUP_EXCLUSIONS: frozenset[str] = frozenset(
    {"ball", "golf_ball", "club_ball", "world"}
)


def _matches(name: str, tokens: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(tok in lower for tok in tokens)


def infer_segment_group_catalog(body_names: Iterable[str]) -> SegmentGroupCatalog:
    """Map model body names into panel groups (legs, pelvis, torso, arms, club)."""
    legs: set[str] = set()
    pelvis: set[str] = set()
    torso: set[str] = set()
    arms: set[str] = set()
    club: set[str] = set()
    for name in body_names:
        if not name:
            continue
        if is_excluded_body_name(name):
            continue
        lower = name.lower()
        if "club" in lower or _matches(name, _CLUB):
            club.add(name)
        elif _matches(name, _LEGS):
            legs.add(name)
        elif "pelvis" in lower or lower in {"hip", "hips"}:
            pelvis.add(name)
        elif _matches(name, _TORSO):
            torso.add(name)
        elif _matches(name, _ARMS):
            arms.add(name)
    return SegmentGroupCatalog(
        legs=frozenset(legs),
        pelvis=frozenset(pelvis),
        torso=frozenset(torso),
        arms=frozenset(arms),
        club=frozenset(club),
    )


def is_excluded_body_name(name: str) -> bool:
    """True when ``name`` is an explicit exclusion (exact body id, not a substring)."""
    return name.lower() in GROUP_EXCLUSIONS


def catalog_member_for_body(catalog: SegmentGroupCatalog, name: str) -> str | None:
    """Return the group key owning ``name``, or ``None`` when unassigned / excluded."""
    if is_excluded_body_name(name):
        return None
    for group, members in catalog.as_mapping().items():
        if name in members:
            return group
    return None
