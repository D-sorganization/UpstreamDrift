"""JSON v2 persistence for body-part visualisation specs.

Defines :class:`SegmentVizSpec` (a single segment's full viz spec — binding,
shape kind & params, fitter kind, theme, visibility) and :class:`SegmentVizSet`
(an ordered, schema-versioned collection).

Schema v2 is a strict superset of the legacy v1 ``SegmentSpec`` shape used in
``segment_set_io``. Loading auto-detects v1 and migrates to v2; saving always
writes v2.

Design-by-Contract
------------------
All public dataclasses are frozen and validate every field in
``__post_init__``. ``shape_params`` are validated per-``shape_kind``: missing
required keys raise ``ValueError`` listing the missing keys; unknown
``shape_kind`` / ``fitter_kind`` raise ``ValueError`` listing the valid
options.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .bindings import BindingKind, MarkerBinding
from .theme import ShapeTheme

__all__ = [
    "SCHEMA_VERSION",
    "SegmentVizSet",
    "SegmentVizSpec",
    "VALID_FITTER_KINDS",
    "VALID_SHAPE_KINDS",
    "migrate_v1_to_v2",
]

SCHEMA_VERSION = 2

VALID_SHAPE_KINDS: tuple[str, ...] = (
    "line",
    "cylinder",
    "ellipsoid",
    "capsule",
    "mesh_file",
    "library_shape",
    "composite",
)

VALID_FITTER_KINDS: tuple[str, ...] = (
    "between_two",
    "cluster_kabsch",
    "procrustes_anisotropic",
)

# Per shape_kind: (required_keys, optional_keys_with_defaults).
# Defaults are applied only when a key is absent so round-trip preserves
# explicit overrides bit-exact.
_SHAPE_PARAM_SPEC: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "line": (("length",), {}),
    "cylinder": (("length", "radius"), {"n_facets": 16}),
    "ellipsoid": (("a", "b", "c"), {"n_lon": 16, "n_lat": 8}),
    "capsule": (("length", "radius"), {"n_facets": 16, "n_lat": 8}),
    "mesh_file": (("path",), {"max_vertices": 5000}),
    "library_shape": (("library_name", "shape_id"), {}),
    "composite": (("children",), {}),
}


def _validate_shape_params(shape_kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise ``shape_params`` for a given ``shape_kind``.

    Returns a fresh dict with defaults filled in for any absent optional keys.
    Raises ``ValueError`` listing missing required keys.
    """
    if not isinstance(params, dict):
        raise TypeError(f"shape_params must be a dict; got {type(params).__name__}")

    required, defaults = _SHAPE_PARAM_SPEC[shape_kind]
    missing = [k for k in required if k not in params]
    if missing:
        raise ValueError(
            f"shape_params for shape_kind={shape_kind!r} missing required "
            f"keys: {missing}"
        )

    out: dict[str, Any] = {}
    # Required keys first, in declared order.
    for key in required:
        out[key] = params[key]
    # Optional keys: explicit value wins; otherwise fill in default.
    for key, default in defaults.items():
        out[key] = params.get(key, default)
    # Preserve any extra keys the caller supplied (e.g. nested composite payloads).
    for key, value in params.items():
        if key not in out:
            out[key] = value

    if shape_kind == "composite":
        children = out["children"]
        if not isinstance(children, list):
            raise ValueError(
                "composite shape_params['children'] must be a list of "
                f"child specs; got {type(children).__name__}"
            )
        normalised_children: list[dict[str, Any]] = []
        for idx, child in enumerate(children):
            if not isinstance(child, dict):
                raise ValueError(
                    f"composite child at index {idx} must be a dict; "
                    f"got {type(child).__name__}"
                )
            child_kind = child.get("shape_kind")
            if child_kind not in VALID_SHAPE_KINDS:
                raise ValueError(
                    f"composite child at index {idx} has unknown "
                    f"shape_kind={child_kind!r}; "
                    f"valid options: {list(VALID_SHAPE_KINDS)}"
                )
            normalised_child_params = _validate_shape_params(
                child_kind, child.get("shape_params", {})
            )
            normalised_children.append(
                {
                    "shape_kind": child_kind,
                    "shape_params": normalised_child_params,
                }
            )
        out["children"] = normalised_children

    return out


def _binding_to_dict(binding: MarkerBinding) -> dict[str, Any]:
    return {
        "kind": binding.kind.value,
        "marker_names": list(binding.marker_names),
        "rest_dimensions": [float(d) for d in binding.rest_dimensions],
        "rest_orientation_quat": [float(c) for c in binding.rest_orientation_quat],
    }


def _binding_from_dict(data: dict[str, Any]) -> MarkerBinding:
    if not isinstance(data, dict):
        raise TypeError(f"binding entry must be a dict; got {type(data).__name__}")
    kind_raw = data.get("kind")
    try:
        kind = BindingKind(kind_raw)
    except ValueError as exc:
        valid = [k.value for k in BindingKind]
        raise ValueError(
            f"unknown binding kind {kind_raw!r}; valid options: {valid}"
        ) from exc
    marker_names = tuple(str(n) for n in data.get("marker_names", ()))
    rest_dimensions = tuple(float(d) for d in data.get("rest_dimensions", ()))
    quat_raw = data.get("rest_orientation_quat", (1.0, 0.0, 0.0, 0.0))
    quat_tuple = tuple(float(c) for c in quat_raw)
    if len(quat_tuple) != 4:
        raise ValueError(
            f"rest_orientation_quat must have exactly 4 entries; got {len(quat_tuple)}"
        )
    return MarkerBinding(
        kind=kind,
        marker_names=marker_names,
        rest_dimensions=rest_dimensions,
        rest_orientation_quat=(
            quat_tuple[0],
            quat_tuple[1],
            quat_tuple[2],
            quat_tuple[3],
        ),
    )


def _theme_to_dict(theme: ShapeTheme) -> dict[str, Any]:
    return {
        "color": theme.color,
        "opacity": float(theme.opacity),
        "edge_color": theme.edge_color,
        "edge_width": float(theme.edge_width),
        "flat_shaded": bool(theme.flat_shaded),
        "group": theme.group,
    }


def _theme_from_dict(data: dict[str, Any]) -> ShapeTheme:
    if not isinstance(data, dict):
        raise TypeError(f"theme entry must be a dict; got {type(data).__name__}")
    return ShapeTheme(
        color=str(data.get("color", "#1f77b4")),
        opacity=float(data.get("opacity", 0.8)),
        edge_color=str(data.get("edge_color", "#000000")),
        edge_width=float(data.get("edge_width", 0.5)),
        flat_shaded=bool(data.get("flat_shaded", True)),
        group=str(data.get("group", "default")),
    )


@dataclass(frozen=True)
class SegmentVizSpec:
    """A single segment's full visualisation spec.

    Attributes
    ----------
    binding:
        How the segment attaches to mocap markers.
    shape_kind:
        One of :data:`VALID_SHAPE_KINDS`.
    shape_params:
        Shape-specific parameters; validated against ``shape_kind``.
    fitter_kind:
        One of :data:`VALID_FITTER_KINDS`.
    theme:
        Visual styling.
    visible:
        Whether the segment is rendered.
    """

    binding: MarkerBinding
    shape_kind: Literal[
        "line",
        "cylinder",
        "ellipsoid",
        "capsule",
        "mesh_file",
        "library_shape",
        "composite",
    ]
    shape_params: dict[str, Any]
    fitter_kind: Literal["between_two", "cluster_kabsch", "procrustes_anisotropic"]
    theme: ShapeTheme
    visible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.binding, MarkerBinding):
            raise TypeError(
                f"binding must be MarkerBinding; got {type(self.binding).__name__}"
            )
        if self.shape_kind not in VALID_SHAPE_KINDS:
            raise ValueError(
                f"unknown shape_kind {self.shape_kind!r}; "
                f"valid options: {list(VALID_SHAPE_KINDS)}"
            )
        if self.fitter_kind not in VALID_FITTER_KINDS:
            raise ValueError(
                f"unknown fitter_kind {self.fitter_kind!r}; "
                f"valid options: {list(VALID_FITTER_KINDS)}"
            )
        if not isinstance(self.theme, ShapeTheme):
            raise TypeError(
                f"theme must be ShapeTheme; got {type(self.theme).__name__}"
            )
        if not isinstance(self.visible, bool):
            raise TypeError(f"visible must be bool; got {type(self.visible).__name__}")
        # Normalise/validate shape_params and reassign on the frozen dc.
        normalised = _validate_shape_params(self.shape_kind, self.shape_params)
        object.__setattr__(self, "shape_params", normalised)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentVizSpec:
        """Construct a :class:`SegmentVizSpec` from a JSON-loaded dict."""
        if not isinstance(data, dict):
            raise TypeError(f"segment entry must be a dict; got {type(data).__name__}")
        shape_kind = data.get("shape_kind")
        if shape_kind not in VALID_SHAPE_KINDS:
            raise ValueError(
                f"unknown shape_kind {shape_kind!r}; "
                f"valid options: {list(VALID_SHAPE_KINDS)}"
            )
        fitter_kind = data.get("fitter_kind")
        if fitter_kind not in VALID_FITTER_KINDS:
            raise ValueError(
                f"unknown fitter_kind {fitter_kind!r}; "
                f"valid options: {list(VALID_FITTER_KINDS)}"
            )
        binding = _binding_from_dict(data.get("binding", {}))
        theme = _theme_from_dict(data.get("theme", {}))
        shape_params = data.get("shape_params", {})
        if not isinstance(shape_params, dict):
            raise TypeError(
                f"shape_params must be a dict; got {type(shape_params).__name__}"
            )
        return cls(
            binding=binding,
            shape_kind=shape_kind,
            shape_params=dict(shape_params),
            fitter_kind=fitter_kind,
            theme=theme,
            visible=bool(data.get("visible", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-ready dict."""
        return {
            "binding": _binding_to_dict(self.binding),
            "shape_kind": self.shape_kind,
            "shape_params": _round_numerics(self.shape_params),
            "fitter_kind": self.fitter_kind,
            "theme": _theme_to_dict(self.theme),
            "visible": bool(self.visible),
        }


def _round_numerics(value: Any) -> Any:
    """Recursively round floats to 1e-9 precision for stable JSON round-trip."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        return round(value, 9)
    if isinstance(value, dict):
        return {k: _round_numerics(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_numerics(v) for v in value]
    if isinstance(value, tuple):
        return [_round_numerics(v) for v in value]
    return value


@dataclass(frozen=True)
class SegmentVizSet:
    """Schema-versioned collection of :class:`SegmentVizSpec`."""

    schema_version: int = SCHEMA_VERSION
    segments: tuple[SegmentVizSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(
            self.schema_version, bool
        ):
            raise TypeError(
                f"schema_version must be int; got {type(self.schema_version).__name__}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"SegmentVizSet only supports schema_version={SCHEMA_VERSION}; "
                f"got {self.schema_version}"
            )
        if not isinstance(self.segments, tuple):
            raise TypeError(
                "segments must be a tuple of SegmentVizSpec; "
                f"got {type(self.segments).__name__}"
            )
        for idx, seg in enumerate(self.segments):
            if not isinstance(seg, SegmentVizSpec):
                raise TypeError(
                    f"segments[{idx}] must be SegmentVizSpec; got {type(seg).__name__}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-ready dict (always v2)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SegmentVizSet:
        """Parse a dict (auto-migrating v1) into a :class:`SegmentVizSet`."""
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be dict; got {type(payload).__name__}")
        version = payload.get("schema_version", SCHEMA_VERSION)
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(f"schema_version must be int; got {version!r}")
        if version == 1:
            payload = migrate_v1_to_v2(payload)
            version = payload["schema_version"]
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version}; "
                f"expected 1 (legacy) or {SCHEMA_VERSION}"
            )
        raw_segments = payload.get("segments", [])
        if not isinstance(raw_segments, list):
            raise ValueError(
                f"'segments' must be a list; got {type(raw_segments).__name__}"
            )
        parsed = tuple(SegmentVizSpec.from_dict(s) for s in raw_segments)
        return cls(schema_version=SCHEMA_VERSION, segments=parsed)

    @classmethod
    def load(cls, path: Path | str) -> SegmentVizSet:
        """Load a viz set from ``path``, auto-migrating v1 payloads."""
        if path is None:
            raise ValueError("path must be provided")
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(f"viz-set file not found: {src}")
        with src.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def save(self, path: Path | str) -> Path:
        """Write this viz set to ``path`` as JSON v2."""
        if path is None:
            raise ValueError("path must be provided")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return out


def migrate_v1_to_v2(v1_dict: dict[str, Any]) -> dict[str, Any]:
    """Migrate a legacy v1 ``SegmentSet`` payload to v2.

    v1 segments had ``a``/``b`` marker names, ``geometry`` in
    ``{"line", "cylinder"}``, ``group``, ``visible``, and ``radius``.
    v2 maps each entry to:

    * ``binding.kind = "between_two"`` with ``marker_names = (a, b)``
    * ``shape_kind = geometry`` (line or cylinder)
    * ``fitter_kind = "between_two"``
    * ``theme = ShapeTheme(group=v1['group'])``

    Cylinder ``radius`` carries over into ``shape_params``; ``length`` is
    left unset on the rest binding (the runtime fitter computes it from
    live marker positions).
    """
    if not isinstance(v1_dict, dict):
        raise TypeError(f"v1_dict must be dict; got {type(v1_dict).__name__}")
    version = v1_dict.get("schema_version", 1)
    if version != 1:
        raise ValueError(f"migrate_v1_to_v2 requires schema_version=1; got {version!r}")
    raw_segments = v1_dict.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError(
            f"v1 'segments' must be a list; got {type(raw_segments).__name__}"
        )

    v2_segments: list[dict[str, Any]] = []
    for entry in raw_segments:
        if not isinstance(entry, dict):
            raise ValueError(
                f"v1 segment entry must be dict; got {type(entry).__name__}"
            )
        a = str(entry["a"])
        b = str(entry["b"])
        geometry = str(entry.get("geometry", "line"))
        if geometry not in ("line", "cylinder"):
            raise ValueError(
                f"v1 geometry must be 'line' or 'cylinder'; got {geometry!r}"
            )
        group = str(entry.get("group", "auto"))
        visible = bool(entry.get("visible", True))
        radius = float(entry.get("radius", 0.015))

        if geometry == "line":
            shape_params: dict[str, Any] = {"length": 1.0}
        else:
            shape_params = {"length": 1.0, "radius": radius, "n_facets": 16}

        v2_segments.append(
            {
                "binding": {
                    "kind": "between_two",
                    "marker_names": [a, b],
                    "rest_dimensions": [],
                    "rest_orientation_quat": [1.0, 0.0, 0.0, 0.0],
                },
                "shape_kind": geometry,
                "shape_params": shape_params,
                "fitter_kind": "between_two",
                "theme": {
                    "color": "#1f77b4",
                    "opacity": 0.8,
                    "edge_color": "#000000",
                    "edge_width": 0.5,
                    "flat_shaded": True,
                    "group": group,
                },
                "visible": visible,
            }
        )

    return {"schema_version": SCHEMA_VERSION, "segments": v2_segments}
