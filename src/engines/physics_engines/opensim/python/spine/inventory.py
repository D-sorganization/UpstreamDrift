"""OpenSim segmental-spine inventory (SPN-8a, #12753).

Maps a model's joints and coordinates to :class:`SpinalLevel` through a
per-model level map (``src/config/spine_model_level_maps.json``), reads each
level's neutral offset ``T_i0`` natively and checks that every rotational
coordinate axis is ISB-aligned.

Conventions (ADR-0055): ISB vertebral frame X anterior, Y cranial, Z right;
coordinate order ``(flex_ext, lat_bend, axial_rot)`` drives the ISB axes
``(Z, X, Y)`` of the ZXY rotation sequence.  ``T_i0`` is the pose of the
SUPERIOR frame in the INFERIOR frame at zero coordinates,

    T_i0 = X_parent_in_base @ inv(X_child_in_base),

with the joint parent frame on the inferior body (translation in metres).
``axis_sign`` is an explicit per-coordinate axis transform: a model axis ``a``
with sign ``s`` is aligned when ``angle(s * a, ISB axis) <= 1 deg``.

Optional per-level translational coordinates (SPN-8c, Senteler-style models)
are ordered ``(antero_posterior, cranio_caudal, medio_lateral)`` and drive the
ISB axes ``(X, Y, Z)`` of the inferior frame (metres); they are checked with
the same 1 deg tolerance and no sign flip.

``observe_native_model`` (tour_matching) reports coordinate and constraint
facts but not joint offset frames or transform axes, so this module adds the
single-hop adapters it needs instead of a second model walker.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    transform_to_matrix,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.levels import SpinalLevel, parse_level

__all__ = [
    "AXIS_TOLERANCE_DEG",
    "DEFAULT_LEVEL_MAP_PATH",
    "ISB_AXES",
    "ISB_TRANSLATION_AXES",
    "LevelMapEntry",
    "ModelLevelMap",
    "SpineInventory",
    "inventory_spine",
    "load_level_maps",
    "neutral_offset_errors",
]

DEFAULT_LEVEL_MAP_PATH = (
    Path(__file__).resolve().parents[6]
    / "src"
    / "config"
    / "spine_model_level_maps.json"
)
AXIS_TOLERANCE_DEG = 1.0
#: ISB axes of (flex_ext, lat_bend, axial_rot): Z right, X anterior, Y cranial.
ISB_AXES = (
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
)
#: ISB axes of (antero_posterior, cranio_caudal, medio_lateral) translations.
ISB_TRANSLATION_AXES = (
    np.array([1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
)


@dataclass(frozen=True)
class LevelMapEntry:
    """Joint and coordinate names (and axis signs) of one level.

    Preconditions: non-empty ``joint``; three distinct non-empty coordinate
    names in (flex_ext, lat_bend, axial_rot) order; ``axis_sign`` in {-1, 1};
    ``translations`` is None or three distinct non-empty names in
    (antero_posterior, cranio_caudal, medio_lateral) order that do not repeat
    a rotational name.
    """

    joint: str
    coordinates: tuple[str, str, str]
    axis_sign: tuple[int, int, int] = (1, 1, 1)
    translations: tuple[str, str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint, str) or not self.joint:
            raise ValueError(f"joint must be a non-empty str, got {self.joint!r}")
        coords = tuple(self.coordinates)
        if len(coords) != 3 or not all(isinstance(c, str) and c for c in coords):
            raise ValueError(f"coordinates must be 3 non-empty names, got {coords!r}")
        if len(set(coords)) != 3:
            raise ValueError(f"coordinates must be distinct, got {coords!r}")
        signs = tuple(self.axis_sign)
        if len(signs) != 3 or any(
            s not in (-1, 1) or isinstance(s, bool) for s in signs
        ):
            raise ValueError(f"axis_sign must be three of +-1, got {signs!r}")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "axis_sign", signs)
        if self.translations is not None:
            trans = tuple(self.translations)
            if len(trans) != 3 or not all(isinstance(c, str) and c for c in trans):
                raise ValueError(
                    f"translations must be 3 non-empty names, got {trans!r}"
                )
            if len(set(trans) | set(coords)) != 6:
                raise ValueError(
                    f"translations must be distinct from each other and from "
                    f"the rotations, got {trans!r}"
                )
            object.__setattr__(self, "translations", trans)


@dataclass(frozen=True)
class ModelLevelMap:
    """Level map of one model id.

    ``verified`` is False for licensed models whose names are not yet
    confirmed against the asset (#12749); such entries carry no levels.
    """

    verified: bool
    pelvis_body: str | None
    terminal_body: str | None
    levels: Mapping[SpinalLevel, LevelMapEntry]
    names_status: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", MappingProxyType(dict(self.levels)))


def _parse_entry(model_id: str, body: Mapping) -> ModelLevelMap:
    levels = {}
    for name, raw in body.get("levels", {}).items():
        try:
            level = parse_level(name)
        except ValueError as exc:
            raise ValueError(f"{model_id}: bad level {name!r}") from exc
        trans = raw.get("translations")
        levels[level] = LevelMapEntry(
            raw["joint"],
            tuple(raw["coordinates"]),
            tuple(raw["axis_sign"]),
            None if trans is None else tuple(trans),
        )
    return ModelLevelMap(
        verified=bool(body["verified"]),
        pelvis_body=body.get("pelvis_body"),
        terminal_body=body.get("terminal_body"),
        levels=levels,
        names_status=str(body.get("names_status", "")),
    )


def load_level_maps(
    path: str | Path = DEFAULT_LEVEL_MAP_PATH,
) -> dict[str, ModelLevelMap]:
    """Load the per-model level maps JSON.

    Raises ``ValueError`` for a bad schema version, level name, coordinate set
    or axis sign.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported level-map schema {raw.get('schema_version')!r}")
    return {mid: _parse_entry(mid, body) for mid, body in raw["models"].items()}


@dataclass(frozen=True)
class SpineInventory:
    """Native inventory of a spine model (caudal-first, contiguous levels).

    ``neutral`` are read-only (4, 4) ``T_i0``; ``axis_alignment_deg`` holds the
    angle (deg) between ``axis_sign * model axis`` and the ISB axis for each
    of (flex_ext, lat_bend, axial_rot); ``axis_failures`` lists levels/axes
    above :data:`AXIS_TOLERANCE_DEG`.  ``translation_names`` and
    ``translation_alignment_deg`` (read-only (3,) angles to the ISB X, Y, Z
    axes) hold only the levels whose map entry declares translational
    coordinates; both are empty for rotation-only maps.
    """

    model_id: str
    levels: tuple[SpinalLevel, ...]
    neutral: Mapping[SpinalLevel, np.ndarray]
    joint_names: Mapping[SpinalLevel, str]
    coordinate_names: Mapping[SpinalLevel, tuple[str, str, str]]
    axis_alignment_deg: Mapping[SpinalLevel, np.ndarray]
    axis_failures: tuple[str, ...] = ()
    translation_names: Mapping[SpinalLevel, tuple[str, str, str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    translation_alignment_deg: Mapping[SpinalLevel, np.ndarray] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def axis_ok(self) -> bool:
        """True when every coordinate axis is ISB-aligned."""
        return not self.axis_failures


# ----------------------------------------------------------- SWIG adapters --


def _joint(model: object, name: str) -> object:
    joints = model.getJointSet()  # type: ignore[attr-defined]
    if joints.getIndex(name) < 0:
        raise ValueError(f"model has no joint {name!r}")
    return joints.get(name)


def _has_body(model: object, name: str) -> bool:
    return model.getBodySet().getIndex(name) >= 0  # type: ignore[attr-defined]


def _joint_coordinate_names(joint: object) -> set[str]:
    return {
        joint.get_coordinates(i).getName()  # type: ignore[attr-defined]
        for i in range(joint.numCoordinates())  # type: ignore[attr-defined]
    }


def _frame_in_base(frame: object) -> np.ndarray:
    return transform_to_matrix(frame.findTransformInBaseFrame())  # type: ignore[attr-defined]


def _neutral_offset(joint: object) -> np.ndarray:
    parent = _frame_in_base(joint.getParentFrame())  # type: ignore[attr-defined]
    child = _frame_in_base(joint.getChildFrame())  # type: ignore[attr-defined]
    return parent @ np.linalg.inv(child)


def _transform_axes(joint: object, first: int) -> dict[str, np.ndarray]:
    """Unit axis of each coordinate driving transform axes ``first..first+2``.

    ``first`` is 0 for rotation1..3 and 3 for translation1..3.
    """
    osim = import_opensim()
    custom = osim.CustomJoint.safeDownCast(joint)
    if custom is None:
        raise ValueError(f"joint {joint.getName()!r} is not a CustomJoint")  # type: ignore[attr-defined]
    spatial = custom.getSpatialTransform()
    out: dict[str, np.ndarray] = {}
    for idx in range(first, first + 3):
        axis = spatial.getTransformAxis(idx)
        names = axis.getCoordinateNames()
        vec = axis.getAxis()
        v = np.array([vec.get(0), vec.get(1), vec.get(2)], dtype=float)
        if names.size() == 1 and np.linalg.norm(v) > 0.0:
            out[names.getValue(0)] = v / np.linalg.norm(v)
    return out


def _rotation_axes(joint: object) -> dict[str, np.ndarray]:
    """Unit axis of each coordinate driving one of rotation1..3."""
    return _transform_axes(joint, 0)


# --------------------------------------------------------------- inventory --


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def _check_map(model: object, entry: ModelLevelMap) -> tuple[SpinalLevel, ...]:
    levels = tuple(sorted(entry.levels, key=lambda lv: lv.ordinal))
    ords = [lv.ordinal for lv in levels]
    if any(b - a != 1 for a, b in zip(ords, ords[1:], strict=False)):
        raise ValueError(
            f"mapped levels must be contiguous, got {[lv.name for lv in levels]}"
        )
    for body in (entry.pelvis_body, entry.terminal_body):
        if body is not None and not _has_body(model, body):
            raise ValueError(f"model has no body {body!r}")
    return levels


def _alignment(
    names: tuple[str, str, str],
    axes: Mapping[str, np.ndarray],
    targets: tuple[np.ndarray, ...],
    signs: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """Read-only (3,) angles (deg) of ``sign * axis`` to each ISB target.

    A coordinate that does not drive a single transform axis gets 180 deg.
    """
    angles = np.full(3, 180.0)
    for k, coord in enumerate(names):
        if coord in axes:
            angles[k] = _angle_deg(signs[k] * axes[coord], targets[k])
    angles.setflags(write=False)
    return angles


def _require_coordinates(
    present: set[str], level: SpinalLevel, lm: LevelMapEntry
) -> None:
    wanted = (*lm.coordinates, *(lm.translations or ()))
    for coord in wanted:
        if coord not in present:
            raise ValueError(
                f"level {level.name}: joint {lm.joint!r} has no coordinate {coord!r}"
            )


def _inventory_level(
    model: object, level: SpinalLevel, lm: LevelMapEntry
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Neutral offset, rotation alignment and (optional) translation alignment."""
    joint = _joint(model, lm.joint)
    _require_coordinates(_joint_coordinate_names(joint), level, lm)
    angles = _alignment(lm.coordinates, _rotation_axes(joint), ISB_AXES, lm.axis_sign)
    trans = None
    if lm.translations is not None:
        trans = _alignment(
            lm.translations, _transform_axes(joint, 3), ISB_TRANSLATION_AXES
        )
    neutral = _neutral_offset(joint)
    neutral.setflags(write=False)
    return neutral, angles, trans


def _failures(
    level: SpinalLevel, names: tuple[str, ...], angles: np.ndarray
) -> list[str]:
    return [
        f"{level.name} {names[k]}: axis {deg:.3f} deg from ISB"
        for k, deg in enumerate(angles)
        if deg > AXIS_TOLERANCE_DEG
    ]


def inventory_spine(
    model_path: str | Path | object,
    model_id: str,
    *,
    level_map: ModelLevelMap | None = None,
    strict: bool = False,
) -> SpineInventory:
    """Inventory the spine joints of a model (a path or a loaded model).

    Preconditions: ``model_id`` names a ``verified`` map (or ``level_map`` is
    given); a path exists; every mapped joint, coordinate and body exists;
    mapped levels are contiguous.  Postconditions: ``neutral`` matrices are
    read-only (4, 4); misaligned axes are listed in ``axis_failures``
    (> 1 deg) and, when ``strict``, raised instead.  Declared translational
    coordinates (optional) are checked like the rotations.

    Raises:
        ValueError: unknown or unverified ``model_id``, missing file, joint,
            coordinate or body, non-contiguous levels, or (strict) a
            misaligned axis.
    """
    entry = level_map
    if entry is None:
        maps = load_level_maps()
        if model_id not in maps:
            raise ValueError(f"unknown model_id {model_id!r}; known {sorted(maps)}")
        entry = maps[model_id]
    if not entry.verified or not entry.levels:
        raise ValueError(
            f"level map for {model_id!r} is UNVERIFIED (names pending asset "
            "confirmation); refusing to inventory"
        )
    if isinstance(model_path, (str, Path)):
        if not Path(model_path).is_file():
            raise ValueError(f"model file not found: {model_path}")
        model = import_opensim().Model(str(model_path))
    else:
        model = model_path
    levels = _check_map(model, entry)
    neutral, angles, trans_angles, failures = {}, {}, {}, []
    trans_names: dict[SpinalLevel, tuple[str, str, str]] = {}
    for level in levels:
        lm = entry.levels[level]
        neutral[level], angles[level], trans = _inventory_level(model, level, lm)
        failures += _failures(level, lm.coordinates, angles[level])
        if trans is not None and lm.translations is not None:
            trans_angles[level] = trans
            trans_names[level] = lm.translations
            failures += _failures(level, lm.translations, trans)
    if strict and failures:
        raise ValueError("axes not ISB-aligned: " + "; ".join(failures))
    return SpineInventory(
        model_id=model_id,
        levels=levels,
        neutral=MappingProxyType(neutral),
        joint_names=MappingProxyType({lv: entry.levels[lv].joint for lv in levels}),
        coordinate_names=MappingProxyType(
            {lv: entry.levels[lv].coordinates for lv in levels}
        ),
        axis_alignment_deg=MappingProxyType(angles),
        axis_failures=tuple(failures),
        translation_names=MappingProxyType(trans_names),
        translation_alignment_deg=MappingProxyType(trans_angles),
    )


def neutral_offset_errors(
    measured: Mapping[SpinalLevel, np.ndarray],
    reference: Mapping[SpinalLevel, np.ndarray],
) -> dict[SpinalLevel, float]:
    """Max-abs difference of (4, 4) neutral offsets per level.

    Raises ``ValueError`` if the level sets differ or a matrix is not (4, 4).
    """
    if set(measured) != set(reference):
        raise ValueError(
            f"levels differ: {sorted(m.name for m in measured)} vs "
            f"{sorted(m.name for m in reference)}"
        )
    out = {}
    for level, mat in measured.items():
        a, b = np.asarray(mat, float), np.asarray(reference[level], float)
        if a.shape != (4, 4) or b.shape != (4, 4):
            raise ValueError(f"{level.name}: offsets must be (4, 4)")
        out[level] = float(np.max(np.abs(a - b)))
    return out
