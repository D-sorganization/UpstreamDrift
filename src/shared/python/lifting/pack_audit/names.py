"""Canonical coordinate names and the per-engine spellings the packs use.

The canonical names come from the vendored ``biomech_parity_standard.json``.
Packs drift from them (``knee_l`` vs ``knee_l_flex``); the tables below are
the audit's measured mapping and are themselves checked against each loaded
model, so a pack rename shows up as an unmapped coordinate, not a silent skip.
"""

from __future__ import annotations

SIDES: tuple[str, str] = ("l", "r")
ENGINES: tuple[str, ...] = ("mujoco", "opensim", "drake", "pinocchio")
LIFTS: tuple[str, ...] = (
    "squat",
    "deadlift",
    "bench_press",
    "snatch",
    "clean_and_jerk",
)

_SIDELESS = ("lumbar_flex", "lumbar_lateral", "lumbar_rotate", "neck_flex")
_SIDED = (
    "shoulder_{s}_flex",
    "shoulder_{s}_adduct",
    "shoulder_{s}_rotate",
    "elbow_{s}_flex",
    "wrist_{s}_flex",
    "wrist_{s}_deviate",
    "hip_{s}_flex",
    "hip_{s}_adduct",
    "hip_{s}_rotate",
    "knee_{s}_flex",
    "ankle_{s}_flex",
    "ankle_{s}_invert",
)

CANONICAL_COORDINATES: tuple[str, ...] = _SIDELESS + tuple(
    pattern.format(s=s) for pattern in _SIDED for s in SIDES
)

_RENAMES: dict[str, dict[str, str]] = {
    "mujoco": {},
    "opensim": {
        "wrist_{s}_deviate": "wrist_{s}_deviation",
        "ankle_{s}_invert": "ankle_{s}_inversion",
    },
    "drake": {
        "knee_{s}_flex": "knee_{s}",
        "elbow_{s}_flex": "elbow_{s}",
        "neck_flex": "neck",
    },
    "pinocchio": {
        "knee_{s}_flex": "knee_{s}",
        "elbow_{s}_flex": "elbow_{s}",
        "neck_flex": "neck",
    },
}


def engine_coordinate_name(engine: str, canonical: str) -> str:
    """Return the name *engine*'s pack uses for the canonical coordinate.

    Raises:
        ValueError: If *engine* or *canonical* is unknown.
    """
    if engine not in _RENAMES:
        raise ValueError(f"unknown engine {engine!r}; use {list(ENGINES)}")
    if canonical not in CANONICAL_COORDINATES:
        raise ValueError(f"unknown canonical coordinate {canonical!r}")
    for pattern, replacement in _RENAMES[engine].items():
        for side in SIDES:
            if canonical == pattern.format(s=side):
                return replacement.format(s=side)
        if "{s}" not in pattern and canonical == pattern:
            return replacement
    return canonical


def canonical_coordinate_name(engine: str, native: str) -> str | None:
    """Inverse of :func:`engine_coordinate_name`; ``None`` when not canonical."""
    for canonical in CANONICAL_COORDINATES:
        if engine_coordinate_name(engine, canonical) == native:
            return canonical
    return None


def expand_phase_key(engine: str, native: str) -> tuple[list[str], bool]:
    """Resolve a phase-objective joint key to canonical coordinate names.

    Returns ``(names, exact)``.  ``exact`` is False when the key had to be
    interpreted (OpenSim's ``hip_flexion`` drives both sides); an empty list
    means the key matches no model coordinate and is reported as unmapped.
    A canonical spelling the pack does not use for its own joints is mapped
    but flagged ``exact=False``.
    """
    exact = canonical_coordinate_name(engine, native)
    if exact is not None:
        return [exact], True
    if native in CANONICAL_COORDINATES:
        # Canonical spelling that is not the pack's own joint name (for
        # example ``knee_l_flex`` against a URDF ``knee_l``): usable, but a
        # name mismatch the report must show.
        return [native], False
    if native.endswith("_flexion"):
        stem = native[: -len("_flexion")]
        sided = [f"{stem}_{s}_flex" for s in SIDES]
        if all(name in CANONICAL_COORDINATES for name in sided):
            return sided, False
        if f"{stem}_flex" in CANONICAL_COORDINATES:
            return [f"{stem}_flex"], False
    return [], False
