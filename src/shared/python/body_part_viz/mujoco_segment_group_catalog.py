"""Heuristic ``SegmentGroupCatalog`` from MuJoCo body names (golf humanoid)."""

from __future__ import annotations

from collections.abc import Iterable

from .segment_angular_momentum import SegmentGroupCatalog

_LEGS = ("thigh", "shank", "foot", "leg", "ankle", "knee", "femur", "tibia", "toe")
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
        lower = name.lower()
        if "club" in lower:
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