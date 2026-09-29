"""JSON persistence for plot-style specifications.

Defines :class:`PlotStyleSpec` (one named entry — usually one per
marker group or trace) and :class:`PlotStyleSet` (an ordered,
schema-versioned collection that round-trips through JSON).

Schema v1 layout::

    {
      "schema_version": 1,
      "entries": [
        {
          "name": "...",
          "target": "...",
          "style": { ...MarkerStyle JSON... }
        },
        ...
      ]
    }

Unknown top-level keys and unknown keys *inside* an entry are tolerated
on load (forward-compat) but never written back.

Note
----
:class:`~src.shared.python.plot_style.markers.CustomMeshSpec` is *not*
JSON-serialisable in v1: persisted MarkerStyles must use a built-in
shape. Serialising custom meshes is tracked for a follow-up issue.

Design-by-Contract
------------------
Both dataclasses are frozen and validate every constraint in
``__post_init__``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from .channels import DataChannel
from .colormaps import ColormapId
from .colors import (
    ColorScale,
    DataDrivenColor,
    PaletteColor,
    StaticColor,
)
from .markers import MarkerShape, MarkerStyle

__all__ = ["PlotStyleSet", "PlotStyleSpec", "SCHEMA_VERSION"]

SCHEMA_VERSION = 1


# ----------------------------------------------------------------------
# JSON helpers (private)
# ----------------------------------------------------------------------


def _color_scale_to_json(scale: ColorScale) -> dict[str, Any]:
    """Serialise a :data:`ColorScale` instance to a JSON-friendly dict."""
    if isinstance(scale, StaticColor):
        return {"kind": "static", "hex_value": scale.hex_value}
    if isinstance(scale, PaletteColor):
        return {
            "kind": "palette",
            "palette_name": scale.palette_name,
            "palette_index": scale.palette_index,
        }
    if isinstance(scale, DataDrivenColor):
        # DataChannel arrays are not JSON-serialisable here; persisted
        # form references the channel by name and unit only. Round-trip
        # tests reconstruct the channel separately.
        return {
            "kind": "data_driven",
            "channel": {
                "name": scale.channel.name,
                "unit": scale.channel.unit,
                "shape": list(scale.channel.values.shape),
                "dtype": str(scale.channel.values.dtype),
            },
            "colormap": scale.colormap.value,
            "vmin": scale.vmin,
            "vmax": scale.vmax,
            "nan_color": scale.nan_color,
        }
    raise TypeError(  # pragma: no cover — guarded by MarkerStyle DbC
        f"unsupported ColorScale variant: {type(scale).__name__}"
    )


def _color_scale_from_json(
    payload: dict[str, Any],
    *,
    channel_lookup: dict[str, DataChannel] | None = None,
) -> ColorScale:
    """Reconstruct a :data:`ColorScale` from JSON.

    For ``data_driven`` entries, the caller may supply
    ``channel_lookup`` mapping channel names to fully-built
    :class:`DataChannel` instances. If the lookup is missing or doesn't
    contain the referenced channel, a placeholder zero-length channel
    with matching name is constructed.
    """
    kind = payload.get("kind")
    if kind == "static":
        return StaticColor(hex_value=str(payload["hex_value"]))
    if kind == "palette":
        return PaletteColor(
            palette_name=str(payload["palette_name"]),
            palette_index=int(payload["palette_index"]),
        )
    if kind == "data_driven":
        ch_payload = cast(dict[str, Any], payload["channel"])
        ch_name = str(ch_payload["name"])
        if channel_lookup is not None and ch_name in channel_lookup:
            channel = channel_lookup[ch_name]
        else:
            import numpy as np

            channel = DataChannel(
                name=ch_name,
                values=np.zeros((0,), dtype=float),
                unit=str(ch_payload.get("unit", "")),
            )
        return DataDrivenColor(
            channel=channel,
            colormap=ColormapId(str(payload["colormap"])),
            vmin=payload.get("vmin"),
            vmax=payload.get("vmax"),
            nan_color=str(payload.get("nan_color", "#888888")),
        )
    raise ValueError(f"unknown ColorScale kind: {kind!r}")


def _marker_style_to_json(style: MarkerStyle) -> dict[str, Any]:
    """Serialise a :class:`MarkerStyle` (without custom mesh) to JSON."""
    if style.shape is MarkerShape.CUSTOM_MESH:
        raise ValueError(
            "MarkerStyle with shape=CUSTOM_MESH is not JSON-serialisable in v1"
        )
    return {
        "shape": style.shape.value,
        "size_px": float(style.size_px),
        "edge_color": style.edge_color,
        "edge_width": float(style.edge_width),
        "fill_color": _color_scale_to_json(style.fill_color),
        "opacity": float(style.opacity),
    }


def _marker_style_from_json(
    payload: dict[str, Any],
    *,
    channel_lookup: dict[str, DataChannel] | None = None,
) -> MarkerStyle:
    """Reconstruct a :class:`MarkerStyle` from JSON."""
    shape = MarkerShape(str(payload["shape"]))
    fill_payload = cast(dict[str, Any], payload["fill_color"])
    fill_color = _color_scale_from_json(fill_payload, channel_lookup=channel_lookup)
    return MarkerStyle(
        shape=shape,
        size_px=float(payload.get("size_px", 6.0)),
        edge_color=str(payload.get("edge_color", "#000000")),
        edge_width=float(payload.get("edge_width", 0.5)),
        fill_color=fill_color,
        opacity=float(payload.get("opacity", 1.0)),
    )


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlotStyleSpec:
    """A single named style entry — usually one per marker group.

    Attributes
    ----------
    name:
        Unique non-empty identifier within an enclosing
        :class:`PlotStyleSet`.
    target:
        Free-form non-empty target descriptor, e.g.
        ``"marker_group:body"`` or ``"trace:clubhead"``.
    style:
        The :class:`MarkerStyle` to apply to ``target``.
    """

    name: str
    target: str
    style: MarkerStyle

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.target, str) or not self.target:
            raise ValueError(f"target must be a non-empty string; got {self.target!r}")
        if not isinstance(self.style, MarkerStyle):
            raise TypeError(
                f"style must be MarkerStyle; got {type(self.style).__name__}"
            )

    def to_json(self) -> dict[str, Any]:
        """Serialise this spec to a JSON-friendly dict."""
        return {
            "name": self.name,
            "target": self.target,
            "style": _marker_style_to_json(self.style),
        }

    @classmethod
    def from_json(
        cls,
        payload: dict[str, Any],
        *,
        channel_lookup: dict[str, DataChannel] | None = None,
    ) -> PlotStyleSpec:
        """Reconstruct a spec from a JSON payload.

        Unknown keys at the top level are tolerated and ignored.
        """
        return cls(
            name=str(payload["name"]),
            target=str(payload["target"]),
            style=_marker_style_from_json(
                cast(dict[str, Any], payload["style"]),
                channel_lookup=channel_lookup,
            ),
        )


@dataclass(frozen=True)
class PlotStyleSet:
    """An ordered, schema-versioned collection of :class:`PlotStyleSpec`.

    Attributes
    ----------
    schema_version:
        Integer schema marker. Currently :data:`SCHEMA_VERSION`.
    entries:
        Tuple of named style entries. Names must be unique within a set.
    """

    schema_version: int = SCHEMA_VERSION
    entries: tuple[PlotStyleSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(
            self.schema_version, bool
        ):
            raise TypeError(
                f"schema_version must be int; got {type(self.schema_version).__name__}"
            )
        if self.schema_version < 1:
            raise ValueError(f"schema_version must be >= 1; got {self.schema_version}")
        if not isinstance(self.entries, tuple):
            raise TypeError(
                "entries must be a tuple of PlotStyleSpec; "
                f"got {type(self.entries).__name__}"
            )
        seen_names: set[str] = set()
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, PlotStyleSpec):
                raise TypeError(
                    f"entries[{index}] must be PlotStyleSpec; "
                    f"got {type(entry).__name__}"
                )
            if entry.name in seen_names:
                raise ValueError(f"duplicate entry name {entry.name!r} in PlotStyleSet")
            seen_names.add(entry.name)

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """Serialise this set to a JSON-friendly dict."""
        return {
            "schema_version": self.schema_version,
            "entries": [entry.to_json() for entry in self.entries],
        }

    @classmethod
    def from_json(
        cls,
        payload: dict[str, Any],
        *,
        channel_lookup: dict[str, DataChannel] | None = None,
    ) -> PlotStyleSet:
        """Reconstruct a set from JSON.

        Unknown keys at any level are tolerated for forward
        compatibility.
        """
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be a dict; got {type(payload).__name__}")
        schema_version = int(payload.get("schema_version", SCHEMA_VERSION))
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise TypeError(
                f"entries must be a JSON list; got {type(raw_entries).__name__}"
            )
        entries = tuple(
            PlotStyleSpec.from_json(
                cast(dict[str, Any], raw_entry),
                channel_lookup=channel_lookup,
            )
            for raw_entry in raw_entries
        )
        return cls(schema_version=schema_version, entries=entries)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        channel_lookup: dict[str, DataChannel] | None = None,
    ) -> PlotStyleSet:
        """Load and parse a :class:`PlotStyleSet` from a JSON file."""
        if not isinstance(path, Path):
            raise TypeError(f"path must be pathlib.Path; got {type(path).__name__}")
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_json(raw, channel_lookup=channel_lookup)

    def save(self, path: Path) -> None:
        """Serialise to JSON and write to ``path``.

        Pretty-prints with two-space indent and a trailing newline.
        """
        if not isinstance(path, Path):
            raise TypeError(f"path must be pathlib.Path; got {type(path).__name__}")
        payload = self.to_json()
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=False)
            handle.write("\n")
