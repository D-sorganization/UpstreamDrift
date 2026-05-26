"""Thin adapter over :mod:`body_part_viz.persistence` for the C3D viewer.

This module historically defined the v1 ``SegmentSpec`` / ``SegmentSet``
dataclasses and a JSON-v1 codec. As of EPIC #4755 the canonical persistence
lives in :class:`body_part_viz.persistence.SegmentVizSet` (schema v2).

The legacy ``SegmentSpec`` / ``SegmentSet`` types remain here for one
release as **deprecated shims** that emit :class:`DeprecationWarning` and
delegate to the v2 layer. ``save_segment_set`` always writes JSON v2;
``load_segment_set`` auto-migrates v1 payloads via
:func:`body_part_viz.persistence.migrate_v1_to_v2`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.python.body_part_viz import (
    SCHEMA_VERSION as _V2_SCHEMA_VERSION,
)
from src.shared.python.body_part_viz import (
    SegmentVizSet,
    SegmentVizSpec,
)

# Public constants kept stable for callers that import them.
SCHEMA_VERSION = _V2_SCHEMA_VERSION
DEFAULT_RADIUS_M = 0.015
_VALID_GEOMETRIES = ("line", "cylinder")

_DEPRECATION_MSG = (
    "{name} is a deprecated v1 shim and will be removed in a follow-up "
    "release; use src.shared.python.body_part_viz.SegmentVizSpec / "
    "SegmentVizSet instead."
)


def _warn_deprecated(name: str) -> None:
    warnings.warn(
        _DEPRECATION_MSG.format(name=name),
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class SegmentSpec:
    """Deprecated v1 shim for a user-editable segment description.

    Validates the same invariants as the original v1 dataclass and exposes
    the same attribute surface (``a``, ``b``, ``geometry``, ``group``,
    ``visible``, ``radius``). New code should construct a
    :class:`body_part_viz.SegmentVizSpec` directly.
    """

    a: str
    b: str
    geometry: str = "line"
    group: str = "auto"
    visible: bool = True
    radius: float = DEFAULT_RADIUS_M

    def __post_init__(self) -> None:
        _warn_deprecated("SegmentSpec")
        if not isinstance(self.a, str) or not self.a:
            raise ValueError(
                f"SegmentSpec.a must be a non-empty string, got {self.a!r}"
            )
        if not isinstance(self.b, str) or not self.b:
            raise ValueError(
                f"SegmentSpec.b must be a non-empty string, got {self.b!r}"
            )
        if self.a == self.b:
            raise ValueError(
                f"SegmentSpec endpoints must differ, got a == b == {self.a!r}"
            )
        if self.geometry not in _VALID_GEOMETRIES:
            raise ValueError(
                f"SegmentSpec.geometry must be one of {_VALID_GEOMETRIES}, "
                f"got {self.geometry!r}"
            )
        if not isinstance(self.group, str) or not self.group:
            raise ValueError(
                f"SegmentSpec.group must be a non-empty string, got {self.group!r}"
            )
        if not isinstance(self.visible, bool):
            raise TypeError(
                f"SegmentSpec.visible must be bool, got {type(self.visible).__name__}"
            )
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, (int, float))
            or self.radius <= 0.0
        ):
            raise ValueError(
                f"SegmentSpec.radius must be a positive number, got {self.radius!r}"
            )


@dataclass
class SegmentSet:
    """Deprecated v1 shim for a persisted set of user segments.

    Wraps a tuple of :class:`SegmentSpec`. Saving / loading goes through
    the v2 :class:`SegmentVizSet` codec, which auto-migrates legacy v1
    payloads on load.
    """

    segments: tuple[SegmentSpec, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _warn_deprecated("SegmentSet")


def default_segment_set_path() -> Path:
    """Return the default JSON path for persisting user segments."""
    return Path.home() / ".golf_modeling_suite" / "c3d_viewer_segments.json"


# ---------------------------------------------------------------- v1 <-> v2


def spec_v1_to_v2(spec: SegmentSpec) -> SegmentVizSpec:
    """Convert a legacy :class:`SegmentSpec` to a :class:`SegmentVizSpec`.

    Deprecated: this helper is a v1 shim. New code should construct
    :class:`~src.shared.python.body_part_viz.SegmentVizSpec` directly
    rather than going through the v1 dataclass layer.
    """
    warnings.warn(
        "spec_v1_to_v2 is deprecated and will be removed in a follow-up "
        "release; construct SegmentVizSpec directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(spec, SegmentSpec):
        raise TypeError(f"spec must be SegmentSpec, got {type(spec).__name__}")
    from src.shared.python.body_part_viz import (
        BindingKind,
        MarkerBinding,
        ShapeTheme,
    )

    binding = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=(spec.a, spec.b),
    )
    shape_kind = spec.geometry  # "line" or "cylinder"
    if shape_kind == "line":
        # Carry v1 radius through as an extra key so v2 -> v1 round-trip is
        # bit-stable; LineShape ignores it but the persistence layer keeps
        # unknown keys verbatim.
        shape_params: dict[str, Any] = {
            "length": 1.0,
            "radius": float(spec.radius),
        }
    else:
        shape_params = {
            "length": 1.0,
            "radius": float(spec.radius),
            "n_facets": 16,
        }
    theme = ShapeTheme(group=spec.group)
    return SegmentVizSpec(
        binding=binding,
        shape_kind=shape_kind,
        shape_params=shape_params,
        fitter_kind="between_two",
        theme=theme,
        visible=bool(spec.visible),
    )


def spec_v2_to_v1(spec: SegmentVizSpec) -> SegmentSpec | None:
    """Convert a :class:`SegmentVizSpec` back to a legacy :class:`SegmentSpec`.

    Returns ``None`` when the v2 spec uses a shape kind that has no v1
    representation (mesh, library, ellipsoid, capsule, composite). Such
    specs survive in the v2 store but are simply not visible to v1-only
    callers.

    Deprecated: this helper is a v1 shim. New code should work with
    :class:`~src.shared.python.body_part_viz.SegmentVizSpec` directly
    rather than converting back to the legacy v1 layer.
    """
    warnings.warn(
        "spec_v2_to_v1 is deprecated and will be removed in a follow-up "
        "release; use SegmentVizSpec directly instead of round-tripping "
        "through the legacy v1 layer.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not isinstance(spec, SegmentVizSpec):
        raise TypeError(f"spec must be SegmentVizSpec, got {type(spec).__name__}")
    if spec.shape_kind not in _VALID_GEOMETRIES:
        return None
    if spec.fitter_kind != "between_two":
        return None
    names = spec.binding.marker_names
    if len(names) != 2:
        return None
    radius = float(spec.shape_params.get("radius", DEFAULT_RADIUS_M))
    return SegmentSpec(
        a=str(names[0]),
        b=str(names[1]),
        geometry=str(spec.shape_kind),
        group=str(spec.theme.group),
        visible=bool(spec.visible),
        radius=radius,
    )


# ------------------------------------------------------------- Public codec


def to_dict(segment_set: SegmentSet | SegmentVizSet) -> dict[str, Any]:
    """Serialise a segment set to a v2 JSON-ready dict."""
    viz = _coerce_to_viz_set(segment_set)
    return viz.to_dict()


def from_dict(payload: dict[str, Any]) -> SegmentSet:
    """Parse a JSON dict into a legacy :class:`SegmentSet` (v1 view).

    Accepts both v1 and v2 payloads. Specs that don't fit the v1
    geometry whitelist (mesh, library, ellipsoid, capsule) are dropped
    from the v1 view; callers that care about those should switch to
    :meth:`SegmentVizSet.from_dict`.
    """
    viz = SegmentVizSet.from_dict(payload)
    v1_specs = tuple(filter(None, (spec_v2_to_v1(s) for s in viz.segments)))
    return SegmentSet(segments=v1_specs, schema_version=SCHEMA_VERSION)


def save_segment_set(path: Path | str, segment_set: SegmentSet | SegmentVizSet) -> Path:
    """Write ``segment_set`` to ``path`` as JSON v2."""
    if path is None:
        raise ValueError("path must be provided")
    viz = _coerce_to_viz_set(segment_set)
    return viz.save(path)


def load_segment_set(path: Path | str) -> SegmentSet:
    """Load a segment set from ``path``, auto-migrating v1 payloads.

    Returns the v1 :class:`SegmentSet` view; for full v2 access use
    :meth:`SegmentVizSet.load`.
    """
    if path is None:
        raise ValueError("path must be provided")
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"segment-set file not found: {src}")
    viz = SegmentVizSet.load(src)
    v1_specs = tuple(filter(None, (spec_v2_to_v1(s) for s in viz.segments)))
    return SegmentSet(segments=v1_specs, schema_version=SCHEMA_VERSION)


def load_viz_set(path: Path | str) -> SegmentVizSet:
    """Load the full v2 :class:`SegmentVizSet` from ``path``.

    This is the preferred entry point for new code that wants access to
    library shapes, mesh files, and other v2-only shape kinds.
    """
    if path is None:
        raise ValueError("path must be provided")
    return SegmentVizSet.load(path)


def _coerce_to_viz_set(segment_set: SegmentSet | SegmentVizSet) -> SegmentVizSet:
    if isinstance(segment_set, SegmentVizSet):
        return segment_set
    if not isinstance(segment_set, SegmentSet):
        raise TypeError(
            "segment_set must be SegmentSet or SegmentVizSet, "
            f"got {type(segment_set).__name__}"
        )
    return SegmentVizSet(
        segments=tuple(spec_v1_to_v2(s) for s in segment_set.segments),
    )


__all__ = [
    "DEFAULT_RADIUS_M",
    "SCHEMA_VERSION",
    "SegmentSet",
    "SegmentSpec",
    "default_segment_set_path",
    "from_dict",
    "load_segment_set",
    "load_viz_set",
    "save_segment_set",
    "spec_v1_to_v2",
    "spec_v2_to_v1",
    "to_dict",
]
