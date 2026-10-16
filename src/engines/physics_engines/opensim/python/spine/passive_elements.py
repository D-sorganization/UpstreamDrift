"""Intervertebral passive elements read from an OpenSim model (SPN-8c, #12755).

Senteler-style spine models carry one bushing force per intervertebral level.
:func:`extract_passive_elements` reads those bushings *at runtime* from the
model's force set into :class:`PassiveElementSpec` records.  Nothing is stored
in the repository: stiffness and damping values of licensed models are never
committed, and the synthetic fixture uses invented numbers
(``fixture_passive``).

Level assignment.  A bushing is assigned to a level through the BODIES that
carry its two frames (``frame.findBaseFrame()``): the level's inferior body is
the base body of the level joint's parent frame and its superior body is the
base body of the child frame (ADR-0055, ``T_i`` maps superior into inferior).
The unordered body pair must equal a mapped level's pair.  Rules, all loud:

* a bushing with at least one frame outside the spine bodies of the mapped
  levels is not a spine element and is ignored;
* a bushing with both frames on the same body, or on two spine bodies that are
  not the pair of a mapped level, raises ``ValueError``;
* two bushings on one level raise ``ValueError`` (ambiguous);
* a bushing class other than ``BushingForce`` and
  ``ExpressionBasedBushingForce`` on a mapped level raises ``ValueError``.

Frame orientation is recorded, never silently swapped: ``frame1_on_inferior``
is True when frame 1 sits on the inferior body.  ``BushingForce`` stiffness
and damping are the diagonal entries (N/m, N m/rad, and their rate terms)
expressed in the axes of frame 1; they are NOT transformed to the ISB
vertebral axes here (the frame axes are model specific).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.engines.physics_engines.opensim.python.spine.inventory import (
    SpineInventory,
    open_model,
)
from src.shared.python.contracts import ensure
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.levels import SpinalLevel

__all__ = ["PassiveElementSpec", "extract_passive_elements"]

_SUPPORTED = ("BushingForce", "ExpressionBasedBushingForce")
_EXPRESSIONS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
_VECTOR_FIELDS = (
    "translational_stiffness",
    "rotational_stiffness",
    "translational_damping",
    "rotational_damping",
)


def _readonly_vec3(value: object, name: str) -> np.ndarray:
    arr = np.array(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and non-negative, got {arr}")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class PassiveElementSpec:
    """One intervertebral bushing as read from a model.

    ``translational_*`` are N/m (stiffness) or N s/m (damping); ``rotational_*``
    are N m/rad or N m s/rad; all are the diagonal in frame-1 axes.  For an
    ``ExpressionBasedBushingForce`` the stiffness vectors are ``None`` and the
    force/moment laws are in ``expressions`` (keys Fx..Mz).

    Preconditions: ``bushing_type`` in the supported classes; vectors finite,
    non-negative and of shape (3,); a ``BushingForce`` has both stiffness
    vectors, an expression-based one a non-empty ``expressions`` mapping.
    Postconditions: vectors are read-only copies; ``expressions`` is a
    read-only mapping.
    """

    level: SpinalLevel
    name: str
    bushing_type: str
    frame1_body: str
    frame2_body: str
    frame1_on_inferior: bool
    translational_stiffness: np.ndarray | None
    rotational_stiffness: np.ndarray | None
    translational_damping: np.ndarray
    rotational_damping: np.ndarray
    expressions: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.bushing_type not in _SUPPORTED:
            raise ValueError(
                f"bushing_type must be one of {_SUPPORTED}, got {self.bushing_type!r}"
            )
        expression_based = self.bushing_type == "ExpressionBasedBushingForce"
        for field_name in _VECTOR_FIELDS:
            value = getattr(self, field_name)
            optional = expression_based and "stiffness" in field_name
            if value is None and optional:
                continue
            if value is None:
                raise ValueError(
                    f"{self.bushing_type} requires stiffness vectors "
                    "(expressions apply only to ExpressionBasedBushingForce)"
                )
            object.__setattr__(self, field_name, _readonly_vec3(value, field_name))
        if expression_based:
            if not self.expressions:
                raise ValueError("ExpressionBasedBushingForce requires expressions")
            object.__setattr__(
                self, "expressions", MappingProxyType(dict(self.expressions))
            )
        elif self.expressions is not None:
            raise ValueError(
                "expressions are only valid for ExpressionBasedBushingForce"
            )


# ----------------------------------------------------------- SWIG adapters --


def _base_name(frame: object) -> str:
    """Name of the body carrying a physical frame."""
    return frame.findBaseFrame().getName()  # type: ignore[attr-defined]


def _joint_bodies(model: object, joint_name: str) -> tuple[str, str]:
    """(inferior, superior) body names of a level joint: parent, child base."""
    joints = model.getJointSet()  # type: ignore[attr-defined]
    if joints.getIndex(joint_name) < 0:
        raise ValueError(f"model has no joint {joint_name!r}")
    joint = joints.get(joint_name)
    return _base_name(joint.getParentFrame()), _base_name(joint.getChildFrame())


def _vec3(vec: object) -> np.ndarray:
    return np.array([vec.get(i) for i in range(3)], dtype=float)  # type: ignore[attr-defined]


def _classify(force: object) -> tuple[str, object] | None:
    """(class name, down-cast force) for a bushing-family force, else None."""
    osim = import_opensim()
    for name in (*_SUPPORTED, "FunctionBasedBushingForce"):
        cast = getattr(osim, name).safeDownCast(force)
        if cast is not None:
            return name, cast
    return None


@dataclass(frozen=True)
class _RawBushing:
    name: str
    kind: str
    force: object
    frame1_body: str
    frame2_body: str


def _bushings(model: object) -> list[_RawBushing]:
    forces = model.getForceSet()  # type: ignore[attr-defined]
    out = []
    for i in range(forces.getSize()):
        found = _classify(forces.get(i))
        if found is None:
            continue
        kind, force = found
        out.append(
            _RawBushing(
                force.getName(),  # type: ignore[attr-defined]
                kind,
                force,
                _base_name(force.getFrame1()),  # type: ignore[attr-defined]
                _base_name(force.getFrame2()),  # type: ignore[attr-defined]
            )
        )
    return out


def _read_damping(force: object) -> tuple[np.ndarray, np.ndarray]:
    return (
        _vec3(force.get_translational_damping()),  # type: ignore[attr-defined]
        _vec3(force.get_rotational_damping()),  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------- extraction --


def _level_pairs(
    model: object, inventory: SpineInventory
) -> dict[frozenset[str], tuple[SpinalLevel, str]]:
    """Unordered body pair -> (level, inferior body) for every inventory level."""
    pairs: dict[frozenset[str], tuple[SpinalLevel, str]] = {}
    for level in inventory.levels:
        inferior, superior = _joint_bodies(model, inventory.joint_names[level])
        key = frozenset((inferior, superior))
        if len(key) != 2 or key in pairs:
            raise ValueError(
                f"level {level.name}: joint bodies {inferior!r}/{superior!r} "
                "do not give a unique body pair"
            )
        pairs[key] = (level, inferior)
    return pairs


def _assign(
    raw: list[_RawBushing], pairs: Mapping[frozenset[str], tuple[SpinalLevel, str]]
) -> dict[SpinalLevel, tuple[_RawBushing, bool]]:
    spine_bodies = set().union(*pairs) if pairs else set()
    assigned: dict[SpinalLevel, tuple[_RawBushing, bool]] = {}
    for item in raw:
        if not {item.frame1_body, item.frame2_body} <= spine_bodies:
            continue  # not a spine element
        key = frozenset((item.frame1_body, item.frame2_body))
        if key not in pairs:
            raise ValueError(
                f"bushing {item.name!r} joins spine bodies {item.frame1_body!r} and "
                f"{item.frame2_body!r}, which are not the body pair of a mapped level"
            )
        level, inferior = pairs[key]
        if level in assigned:
            raise ValueError(
                f"ambiguous passive element for level {level.name}: bushings "
                f"{assigned[level][0].name!r} and {item.name!r}"
            )
        assigned[level] = (item, item.frame1_body == inferior)
    return assigned


def _to_spec(
    level: SpinalLevel, item: _RawBushing, on_inferior: bool
) -> PassiveElementSpec:
    if item.kind not in _SUPPORTED:
        raise ValueError(
            f"level {level.name}: bushing {item.name!r} is a {item.kind}, which is "
            f"not supported (supported: {_SUPPORTED})"
        )
    t_damp, r_damp = _read_damping(item.force)
    if item.kind == "BushingForce":
        t_stiff: np.ndarray | None = _vec3(
            item.force.get_translational_stiffness()  # type: ignore[attr-defined]
        )
        r_stiff: np.ndarray | None = _vec3(
            item.force.get_rotational_stiffness()  # type: ignore[attr-defined]
        )
        expressions = None
    else:
        t_stiff = r_stiff = None
        expressions = {
            key: getattr(item.force, f"get_{key}_expression")() for key in _EXPRESSIONS
        }
    return PassiveElementSpec(
        level=level,
        name=item.name,
        bushing_type=item.kind,
        frame1_body=item.frame1_body,
        frame2_body=item.frame2_body,
        frame1_on_inferior=on_inferior,
        translational_stiffness=t_stiff,
        rotational_stiffness=r_stiff,
        translational_damping=t_damp,
        rotational_damping=r_damp,
        expressions=expressions,
    )


def extract_passive_elements(
    model_path: str | Path | object,
    inventory: SpineInventory,
    *,
    require_all: bool = False,
) -> Mapping[SpinalLevel, PassiveElementSpec]:
    """Read the intervertebral bushings of a model, one per inventory level.

    Args:
        model_path: a model file or an already loaded ``opensim.Model``.
        inventory: the :class:`SpineInventory` of that model; its joints define
            each level's inferior and superior body (see module docstring for
            the assignment rule).
        require_all: raise if a level of the inventory has no bushing.

    Preconditions: ``inventory`` is a :class:`SpineInventory`; a path exists;
    every inventory joint exists in the model.  Postconditions: keys follow
    ``inventory.levels`` order and are a subset of it (all of it when
    ``require_all``); every value's vectors are read-only; values are read at
    runtime from the model and never cached or committed.

    Raises:
        TypeError: ``inventory`` is not a :class:`SpineInventory`.
        ValueError: missing file or joint; ambiguous, unmatched or unsupported
            bushings (see module docstring); a missing level with
            ``require_all``.
    """
    if not isinstance(inventory, SpineInventory):
        raise TypeError(
            f"inventory must be a SpineInventory, got {type(inventory).__name__}"
        )
    model = open_model(model_path)
    pairs = _level_pairs(model, inventory)
    assigned = _assign(_bushings(model), pairs)
    missing = [lv.name for lv in inventory.levels if lv not in assigned]
    if require_all and missing:
        raise ValueError(f"no bushing found for levels {missing}")
    out = {
        level: _to_spec(level, *assigned[level])
        for level in inventory.levels
        if level in assigned
    }
    ensure(
        all(
            math.isfinite(float(np.sum(s.translational_damping))) for s in out.values()
        ),
        "extracted damping must be finite",
    )
    return MappingProxyType(out)
