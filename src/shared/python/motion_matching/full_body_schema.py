"""One registry for the full-body specification schemas (HRG-D1, #13170).

Every consumer that depends on the schema of a full-body document dispatches
here, on the document's ``schema_version``, instead of comparing strings in
place. The registry is declarative: each schema id maps to the module that
provides it, and that module exposes one :class:`FullBodySchema` named
``FULL_BODY_SCHEMA``. Provider modules are imported lazily, so the registry
has no import-time dependency on any of them (no cycles).

Schemas:

* ``full-body-v1`` (the default): the qualified upper body plus lower limbs;
  the trunk is ``TorsoInput`` (twist about pelvis z) and the
  ``SpineInputX/Y`` universal joint at one L5/S1 pivot.
* ``full-body-v2``: the same document with the trunk replaced by two
  regional lumped 3-DOF joints labelled at L5/S1 (lumbar) and T12/L1
  (thoracic); see :mod:`full_body_v2`.

The file name ``full_body_spec_v2.json`` is a *revision* of a v1 document and
has nothing to do with the ``full-body-v2`` schema.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from src.shared.python.core.error_utils import ModelError

__all__ = [
    "DEFAULT_FULL_BODY_SCHEMA",
    "ISB_SPINE_AXES",
    "FullBodySchema",
    "SchemaFeatureUnavailableError",
    "TrunkJointDefinition",
    "TrunkTopology",
    "full_body_schema",
    "require_full_body_schema",
    "supported_full_body_schemas",
]

DEFAULT_FULL_BODY_SCHEMA = "full-body-v1"
#: ISB ``isb_spine`` (ZXY) angle names, in sequence order (ADR-0055 item 1).
ISB_SPINE_AXES = ("flexion_extension", "lateral_bending", "axial_rotation")
_PRIMITIVES = ("Rx", "Ry", "Rz")
_PROVIDERS: Mapping[str, str] = MappingProxyType(
    {
        "full-body-v1": "src.shared.python.motion_matching.full_body_spec",
        "full-body-v2": "src.shared.python.motion_matching.full_body_v2",
    }
)


class SchemaFeatureUnavailableError(ModelError, NotImplementedError):
    """A consumer feature is not yet provided for a registered schema.

    The schema itself is supported; one feature of one consumer is delivered
    by a later, tracked HRG child. The consumer fails closed instead of
    returning a silently wrong result. ``schema_id``, ``feature``, ``child``
    and ``issue`` are kept as attributes; the message names all of them.
    """

    def __init__(
        self,
        schema_id: str,
        *,
        consumer: str,
        feature: str,
        child: str,
        issue: int,
    ) -> None:
        if not isinstance(issue, int) or issue <= 0:
            raise ValueError(f"issue must be a positive issue number, got {issue!r}")
        if not child or not feature:
            raise ValueError("child and feature must be non-empty")
        self.schema_id = schema_id
        self.feature = feature
        self.child = child
        self.issue = issue
        super().__init__(
            schema_id,
            consumer,
            f"{feature} is not available for {schema_id} until {child} "
            f"(tracked: #{issue})",
        )


@dataclass(frozen=True)
class TrunkJointDefinition:
    """One trunk joint of a schema, declared about model axes.

    ``primitives[k]`` drives ``coordinates[k]``. The ISB ``isb_spine`` angle of
    that coordinate is ``isb_signs[k] * value`` about ``isb_axes[k]``.
    ``soft_limits_isb_deg`` (optional) are per-coordinate ``(lo, hi)`` in ISB
    sign (extension, right lateral bending and left axial rotation positive).

    Preconditions (``ValueError``): non-empty names; rotational primitives
    only; one unique coordinate, ISB axis and unit sign per primitive; limits,
    when given, are finite with ``lo < hi``.
    """

    name: str
    level: str
    parent_role: str
    child_role: str
    primitives: tuple[str, ...]
    coordinates: tuple[str, ...]
    isb_axes: tuple[str, ...]
    isb_signs: tuple[float, ...]
    soft_limits_isb_deg: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        for label in ("name", "level", "parent_role", "child_role"):
            if not str(getattr(self, label)).strip():
                raise ValueError(f"Trunk joint {label} is required")
        n = len(self.primitives)
        if n == 0 or any(p not in _PRIMITIVES for p in self.primitives):
            raise ValueError("Trunk joint primitives must be Rx, Ry or Rz")
        lengths = {len(self.coordinates), len(self.isb_axes), len(self.isb_signs)}
        if lengths != {n} or len(set(self.coordinates)) != n:
            raise ValueError("One unique coordinate, ISB axis and sign per primitive")
        if any(a not in ISB_SPINE_AXES for a in self.isb_axes):
            raise ValueError(f"ISB axes must be from {ISB_SPINE_AXES}")
        if any(s not in (-1.0, 1.0) for s in self.isb_signs):
            raise ValueError("ISB signs must be +1 or -1")
        limits = self.soft_limits_isb_deg
        if limits is not None and (
            len(limits) != n or any(not float(lo) < float(hi) for lo, hi in limits)
        ):
            raise ValueError("Soft limits need one ordered (lo, hi) per coordinate")


@dataclass(frozen=True)
class TrunkTopology:
    """The trunk of a schema: its joints and coordinates in document order."""

    joints: tuple[TrunkJointDefinition, ...]
    coordinates: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        declared = tuple(c for j in self.joints for c in j.coordinates)
        if sorted(declared) != sorted(self.coordinates) or len(set(declared)) != len(
            declared
        ):
            raise ValueError(
                "Trunk coordinates must be exactly the joints' coordinates"
            )
        if not self.description.strip():
            raise ValueError("Trunk description is required")

    def isb_axis(self, coordinate: str) -> tuple[str, float]:
        """Return ``(isb_spine axis name, sign)`` of a trunk coordinate.

        Raises ``ValueError`` for a coordinate that is not part of the trunk.
        """
        for joint in self.joints:
            if coordinate in joint.coordinates:
                k = joint.coordinates.index(coordinate)
                return joint.isb_axes[k], float(joint.isb_signs[k])
        raise ValueError(f"{coordinate!r} is not a trunk coordinate")

    def locate(self, spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """The spec's joint documents of this trunk, in declaration order.

        A joint matches a definition when its primitive sequence and
        coordinates are exactly the declared ones. Raises ``ValueError``
        (mentioning the trunk) when a joint is missing or ambiguous, or when
        consecutive joints are not parent and child.
        """
        found = []
        for joint_def in self.joints:
            declared = list(
                zip(joint_def.primitives, joint_def.coordinates, strict=True)
            )
            matches = [
                j
                for j in spec.get("joints", ())
                if [(p["primitive"], p["coordinate"]) for p in j["primitives"]]
                == declared
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one trunk joint {joint_def.name!r} with "
                    f"{declared}, found {len(matches)}"
                )
            found.append(matches[0])
        for parent, child in zip(found, found[1:], strict=False):
            if child["parent"] != parent["child"]:
                raise ValueError(
                    f"Trunk joint {child['name']!r} is not a child of {parent['name']!r}"
                )
        return tuple(found)


@dataclass(frozen=True)
class FullBodySchema:
    """A registered schema: id, declared trunk and its document validator.

    ``validate(document, upper)`` raises ``ValueError`` unless ``document`` is
    a valid document of this schema whose upper-body slice is ``upper``.
    """

    schema_id: str
    trunk: TrunkTopology
    validate: Callable[[Mapping[str, Any], Mapping[str, Any]], None]
    description: str


def supported_full_body_schemas() -> tuple[str, ...]:
    """Registered schema ids, oldest first; the first is the default."""
    return tuple(_PROVIDERS)


def _schema_id(spec_or_id: Mapping[str, Any] | str) -> object:
    if isinstance(spec_or_id, str):
        return spec_or_id
    if isinstance(spec_or_id, Mapping):
        return spec_or_id.get("schema_version")
    raise TypeError("Expected a full-body document Mapping or a schema id string")


def full_body_schema(spec_or_id: Mapping[str, Any] | str) -> FullBodySchema:
    """Resolve the registered schema of a document (or of a schema id).

    Raises ``ValueError`` (``Unsupported schema version ...``) for an
    unregistered id and ``TypeError`` for an argument that is neither.
    Postcondition: the returned schema's ``schema_id`` equals the requested id.
    """
    schema_id = _schema_id(spec_or_id)
    if not isinstance(schema_id, str) or schema_id not in _PROVIDERS:
        raise ValueError(
            f"Unsupported schema version {schema_id!r}: expected one of "
            f"{', '.join(_PROVIDERS)}"
        )
    schema = importlib.import_module(_PROVIDERS[schema_id]).FULL_BODY_SCHEMA
    if not isinstance(schema, FullBodySchema) or schema.schema_id != schema_id:
        raise ValueError(f"Provider of {schema_id!r} does not declare that schema")
    return schema


def require_full_body_schema(
    spec: Mapping[str, Any],
    *,
    consumer: str,
    accepted: Sequence[str] | None = None,
) -> FullBodySchema:
    """Return the schema of ``spec`` if ``consumer`` supports it.

    ``accepted`` restricts the registered schemas a consumer handles (all of
    them by default). Raises ``TypeError`` unless ``spec`` is a Mapping, and
    ``ValueError`` naming the consumer and the accepted ids otherwise.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"{consumer}: the full-body spec must be a Mapping")
    allowed = supported_full_body_schemas() if accepted is None else tuple(accepted)
    unknown = [a for a in allowed if a not in _PROVIDERS]
    if not allowed or unknown:
        raise ValueError(f"{consumer}: accepted schemas {unknown} are not registered")
    schema_id = spec.get("schema_version")
    if schema_id not in allowed:
        raise ValueError(
            f"Unsupported schema version {schema_id!r} for {consumer}: "
            f"expected one of {', '.join(allowed)}"
        )
    return full_body_schema(spec)
