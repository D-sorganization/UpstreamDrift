"""Golf-humanoid to full-body coordinate map (SPN-9c, #12758).

``src/config/spine_fullbody_coordinate_maps.json`` records, per full-body model
id, how every ``golf_humanoid`` coordinate drives the full-body model and what
happens to every full-body coordinate.  The file stores names, joint topology,
signs and treatments only; no value, offset, mass or geometry of a licensed
model.

Treatments of a golf coordinate:

``direct``
    ``target = sign * golf`` (one to one, no offset).  ``sign`` is the observed
    direction of the target axis relative to the golf axis at the all-zero
    pose; it is checked natively (``probe_axis_dots``), never assumed.
``pelvis_pose``
    carried by the pelvis pose of the golf input (same ground-pelvis encoding
    in both models); the sign is checked, not applied.
``orientation_match``
    the golf and target coordinates parametrise one segment orientation with
    different Euler sequences; solved per frame natively (see
    :mod:`fullbody_native`), grouped in ``orientation_groups``.
``spine_drive``
    consumed by the segmental spine drive as the pelvis-to-torso rotation.
``excluded``
    no driven target; a reason is recorded.

Treatments of a target coordinate: ``driven_by_pelvis_pose``,
``driven_by_spine_drive``, ``driven_direct``, ``driven_orientation_match``,
``held_neutral`` (free in the asset, prescribed at zero) and
``locked_in_asset`` (locked in the file, never driven).

Every coordinate of both models must be classified exactly once
(:meth:`FullbodyMap.check_coverage`); an unclassified or missing name is an
error, never silently dropped.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_FULLBODY_MAP_PATH",
    "GOLF_TREATMENTS",
    "SCHEMA_VERSION",
    "TARGET_TREATMENTS",
    "DirectEntry",
    "FullbodyMap",
    "OrientationGroup",
    "apply_direct_map",
    "get_fullbody_map",
    "load_fullbody_maps",
    "neutral_columns",
    "parse_fullbody_maps",
]

DEFAULT_FULLBODY_MAP_PATH = (
    Path(__file__).resolve().parents[6]
    / "src"
    / "config"
    / "spine_fullbody_coordinate_maps.json"
)
SCHEMA_VERSION = 1
GOLF_TREATMENTS = (
    "direct",
    "pelvis_pose",
    "orientation_match",
    "spine_drive",
    "excluded",
)
TARGET_TREATMENTS = (
    "driven_by_pelvis_pose",
    "driven_by_spine_drive",
    "driven_direct",
    "driven_orientation_match",
    "held_neutral",
    "locked_in_asset",
)
_NEEDS_REASON = ("excluded", "spine_drive")
_AXIS_DOT_FLOOR = 0.5


def _name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string, got {value!r}")
    return value


def _three(value: object, label: str) -> tuple[str, str, str]:
    items = tuple(value) if isinstance(value, list | tuple) else ()
    if len(items) != 3:
        raise ValueError(f"{label} must list exactly three names, got {value!r}")
    names = tuple(_name(item, label) for item in items)
    if len(set(names)) != 3:
        raise ValueError(f"{label} must be distinct, got {names!r}")
    return names[0], names[1], names[2]


@dataclass(frozen=True)
class DirectEntry:
    """One golf coordinate carried to one target coordinate with a sign.

    Preconditions (``ValueError``): non-empty names, ``sign`` in {-1, 1},
    ``min_axis_dot`` in [0.5, 1] (the smallest accepted ``sign`` times the
    cosine between the golf and target axes).
    """

    golf: str
    target: str
    sign: int
    min_axis_dot: float

    def __post_init__(self) -> None:
        _name(self.golf, "golf")
        _name(self.target, "target")
        if isinstance(self.sign, bool) or self.sign not in (-1, 1):
            raise ValueError(
                f"sign of {self.golf!r} must be -1 or 1, got {self.sign!r}"
            )
        dot = self.min_axis_dot
        if (
            isinstance(dot, bool)
            or not isinstance(dot, int | float)
            or not _AXIS_DOT_FLOOR <= dot <= 1.0
        ):
            raise ValueError(
                f"min_axis_dot of {self.golf!r} must lie in "
                f"[{_AXIS_DOT_FLOOR}, 1], got {dot!r}"
            )


@dataclass(frozen=True)
class OrientationGroup:
    """Golf and target coordinates that parametrise one segment orientation.

    ``golf_coordinates``/``target_coordinates`` are listed in joint order;
    ``golf_body``/``target_body`` are the moving segments whose world
    orientations are matched.
    """

    group_id: str
    golf_body: str
    golf_coordinates: tuple[str, str, str]
    target_body: str
    target_coordinates: tuple[str, str, str]

    def __post_init__(self) -> None:
        _name(self.group_id, "group_id")
        _name(self.golf_body, "golf_body")
        _name(self.target_body, "target_body")
        object.__setattr__(
            self, "golf_coordinates", _three(self.golf_coordinates, "golf_coordinates")
        )
        object.__setattr__(
            self,
            "target_coordinates",
            _three(self.target_coordinates, "target_coordinates"),
        )


def _frozen_map(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class FullbodyMap:
    """The classified coordinate map of one full-body model (frozen).

    ``golf_treatment``/``target_treatment`` give the treatment of every
    coordinate; ``golf_reason`` the recorded reason of every excluded or
    spine-drive coordinate.  ``hand_bodies`` maps golf hand bodies to the
    target hand bodies that receive the club wrench; ``foot_bodies`` are the
    target bodies carrying the ground-reaction sole points.
    """

    map_id: str
    verified: bool
    names_status: str
    direct: tuple[DirectEntry, ...]
    pelvis: tuple[DirectEntry, ...]
    groups: tuple[OrientationGroup, ...]
    golf_treatment: Mapping[str, str]
    golf_reason: Mapping[str, str]
    target_treatment: Mapping[str, str]
    hand_bodies: Mapping[str, str]
    foot_bodies: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        for attr in (
            "golf_treatment",
            "golf_reason",
            "target_treatment",
            "hand_bodies",
        ):
            object.__setattr__(self, attr, _frozen_map(getattr(self, attr)))

    def targets_with(self, treatment: str) -> tuple[str, ...]:
        """Target coordinate names with ``treatment`` (map order)."""
        if treatment not in TARGET_TREATMENTS:
            raise ValueError(f"unknown target treatment {treatment!r}")
        return tuple(n for n, t in self.target_treatment.items() if t == treatment)

    def golf_with(self, treatment: str) -> tuple[str, ...]:
        """Golf coordinate names with ``treatment`` (map order)."""
        if treatment not in GOLF_TREATMENTS:
            raise ValueError(f"unknown golf treatment {treatment!r}")
        return tuple(n for n, t in self.golf_treatment.items() if t == treatment)

    def check_coverage(
        self, golf_names: Iterable[str], target_names: Iterable[str]
    ) -> None:
        """Require the map to classify exactly the given coordinate sets.

        Raises:
            ValueError: a coordinate of either model is unclassified, or a
                classified name is absent from the model it names.
        """
        for label, classified, observed in (
            ("golf", set(self.golf_treatment), set(golf_names)),
            ("target", set(self.target_treatment), set(target_names)),
        ):
            extra = sorted(observed - classified)
            if extra:
                raise ValueError(f"unclassified {label} coordinates: {extra}")
            missing = sorted(classified - observed)
            if missing:
                raise ValueError(
                    f"classified {label} coordinates absent from the {label} "
                    f"model: {missing}"
                )

    def summary(self) -> dict[str, Any]:
        """JSON-compatible statement of what is and is not driven.

        Lists the golf coordinates by treatment (with reasons for the excluded
        ones) and counts the target coordinates by treatment; used in receipts.
        """
        golf = {t: sorted(self.golf_with(t)) for t in GOLF_TREATMENTS}
        counts = Counter(self.target_treatment.values())
        return {
            "map_id": self.map_id,
            "golf_coordinates": golf,
            "excluded_reasons": {n: self.golf_reason[n] for n in golf["excluded"]},
            "target_counts": {t: counts.get(t, 0) for t in TARGET_TREATMENTS},
            "held_neutral": sorted(self.targets_with("held_neutral")),
        }


# ------------------------------------------------------------------- parsing


def _parse_golf_rows(
    rows: object, groups: Mapping[str, OrientationGroup]
) -> tuple[list[DirectEntry], list[DirectEntry], dict[str, str], dict[str, str]]:
    if not isinstance(rows, list):
        raise ValueError("golf_coordinates must be a list")
    direct: list[DirectEntry] = []
    pelvis: list[DirectEntry] = []
    treatment: dict[str, str] = {}
    reason: dict[str, str] = {}
    for row in rows:
        name = _name(row.get("golf"), "golf")
        kind = row.get("treatment")
        if kind not in GOLF_TREATMENTS:
            raise ValueError(f"unknown golf treatment {kind!r} for {name!r}")
        if name in treatment:
            raise ValueError(f"golf coordinate {name!r} is listed twice")
        treatment[name] = kind
        if kind in ("direct", "pelvis_pose"):
            entry = DirectEntry(
                name, row.get("target"), row.get("sign"), row.get("min_axis_dot")
            )
            (direct if kind == "direct" else pelvis).append(entry)
        if kind in _NEEDS_REASON:
            reason[name] = _name(row.get("reason"), f"reason of {name}")
        if kind == "orientation_match" and row.get("group") not in groups:
            raise ValueError(f"{name!r} names an unknown orientation group")
    return direct, pelvis, treatment, reason


def _parse_targets(block: object) -> dict[str, str]:
    if not isinstance(block, Mapping):
        raise ValueError("target_coordinates must be a mapping")
    if set(block) != set(TARGET_TREATMENTS):
        raise ValueError(
            f"target_coordinates must list exactly the treatments "
            f"{TARGET_TREATMENTS}, got {sorted(block)}"
        )
    out: dict[str, str] = {}
    for kind in TARGET_TREATMENTS:
        for name in block[kind]:
            if name in out:
                raise ValueError(f"target coordinate {name!r} is classified twice")
            out[_name(name, "target coordinate")] = kind
    return out


def _parse_groups(block: object) -> dict[str, OrientationGroup]:
    if not isinstance(block, Mapping):
        raise ValueError("orientation_groups must be a mapping")
    return {
        gid: OrientationGroup(
            gid,
            body["golf_body"],
            body["golf_coordinates"],
            body["target_body"],
            body["target_coordinates"],
        )
        for gid, body in block.items()
    }


def _check_consistency(
    direct: list[DirectEntry],
    pelvis: list[DirectEntry],
    groups: Mapping[str, OrientationGroup],
    golf: Mapping[str, str],
    target: Mapping[str, str],
) -> None:
    seen: set[str] = set()
    for entries, wanted in (
        (direct, "driven_direct"),
        (pelvis, "driven_by_pelvis_pose"),
    ):
        for entry in entries:
            if entry.target in seen:
                raise ValueError(f"duplicate target {entry.target!r}")
            seen.add(entry.target)
            if target.get(entry.target) != wanted:
                raise ValueError(
                    f"target {entry.target!r} of {entry.golf!r} is not classified "
                    f"{wanted!r}"
                )
    for group in groups.values():
        for name in group.target_coordinates:
            if name in seen:
                raise ValueError(f"duplicate target {name!r}")
            seen.add(name)
            if target.get(name) != "driven_orientation_match":
                raise ValueError(
                    f"target {name!r} of group {group.group_id!r} is not "
                    f"classified 'driven_orientation_match'"
                )
        for name in group.golf_coordinates:
            if golf.get(name) != "orientation_match":
                raise ValueError(
                    f"golf {name!r} of group {group.group_id!r} is not "
                    f"'orientation_match'"
                )
    for kind in ("driven_direct", "driven_by_pelvis_pose", "driven_orientation_match"):
        unmatched = sorted(n for n, t in target.items() if t == kind and n not in seen)
        if unmatched:
            raise ValueError(f"{kind} targets with no golf source: {unmatched}")


def _parse_one(map_id: str, body: Mapping[str, Any]) -> FullbodyMap:
    groups = _parse_groups(body.get("orientation_groups", {}))
    direct, pelvis, golf, reason = _parse_golf_rows(
        body.get("golf_coordinates"), groups
    )
    target = _parse_targets(body.get("target_coordinates"))
    _check_consistency(direct, pelvis, groups, golf, target)
    hands = body.get("hand_bodies", {})
    if not isinstance(hands, Mapping) or not hands:
        raise ValueError("hand_bodies must be a non-empty mapping")
    for key, value in hands.items():
        _name(key, "hand_bodies key")
        _name(value, f"hand_bodies[{key!r}]")
    feet = tuple(_name(n, "foot_bodies") for n in body.get("foot_bodies", ()))
    return FullbodyMap(
        map_id=map_id,
        verified=bool(body.get("verified", False)),
        names_status=str(body.get("names_status", "")),
        direct=tuple(direct),
        pelvis=tuple(pelvis),
        groups=tuple(groups.values()),
        golf_treatment=golf,
        golf_reason=reason,
        target_treatment=target,
        hand_bodies=dict(hands),
        foot_bodies=feet,
    )


def parse_fullbody_maps(data: Mapping[str, Any]) -> Mapping[str, FullbodyMap]:
    """Validate a parsed JSON document and return its maps by id.

    Raises:
        TypeError: ``data`` is not a mapping.
        ValueError: unsupported ``schema_version``, an unknown treatment, a
            duplicate or unclassified target, an inconsistent sign or group,
            or a missing reason.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"data must be a mapping, got {type(data).__name__}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {data.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    maps = data.get("maps")
    if not isinstance(maps, Mapping) or not maps:
        raise ValueError("maps must be a non-empty mapping")
    return MappingProxyType({mid: _parse_one(mid, body) for mid, body in maps.items()})


def load_fullbody_maps(
    path: str | Path = DEFAULT_FULLBODY_MAP_PATH,
) -> Mapping[str, FullbodyMap]:
    """Read and validate the map file (default: the committed registry).

    Raises:
        ValueError: the file is absent, not JSON, or fails
            :func:`parse_fullbody_maps`.
    """
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"full-body map file not found: {source.name}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"full-body map file is not valid JSON: {exc}") from exc
    return parse_fullbody_maps(raw)


def get_fullbody_map(
    map_id: str, path: str | Path = DEFAULT_FULLBODY_MAP_PATH
) -> FullbodyMap:
    """Return the map named ``map_id``.

    Raises:
        TypeError: ``map_id`` is not a string.
        ValueError: unknown id (the known ids are listed).
    """
    if not isinstance(map_id, str):
        raise TypeError(f"map_id must be a str, got {type(map_id).__name__}")
    maps = load_fullbody_maps(path)
    if map_id not in maps:
        raise ValueError(f"unknown map id {map_id!r}; known {sorted(maps)}")
    return maps[map_id]


# ------------------------------------------------------------------ transfer


def apply_direct_map(
    fmap: FullbodyMap, golf_values: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """``sign * golf`` for every direct entry present in ``golf_values``.

    Postcondition: new arrays; entries whose golf coordinate is absent from
    ``golf_values`` are omitted (the caller decides how to default them).
    Raises ``ValueError`` for a non-finite series.
    """
    out: dict[str, np.ndarray] = {}
    for entry in fmap.direct:
        if entry.golf not in golf_values:
            continue
        series = np.array(golf_values[entry.golf], dtype=float, copy=True)
        if not np.all(np.isfinite(series)):
            raise ValueError(f"golf coordinate {entry.golf!r} has non-finite values")
        out[entry.target] = entry.sign * series
    return out


def neutral_columns(fmap: FullbodyMap, n_frames: int) -> dict[str, np.ndarray]:
    """Zero series for every ``held_neutral`` target coordinate.

    Raises ``ValueError`` unless ``n_frames`` is a positive integer.
    """
    if isinstance(n_frames, bool) or not isinstance(n_frames, int) or n_frames < 1:
        raise ValueError(f"n_frames must be a positive integer, got {n_frames!r}")
    return {n: np.zeros(n_frames) for n in fmap.targets_with("held_neutral")}
