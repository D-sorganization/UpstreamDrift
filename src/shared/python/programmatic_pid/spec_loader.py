"""Spec loading, accessor utilities, and configuration extraction.

DRY: The SpecAccessor class replaces the scattered get_project / get_drawing /
get_text_config / get_layout_config / get_layer_config accessor functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from programmatic_pid.geometry import to_float
from programmatic_pid.profiles import apply_profile
from programmatic_pid.types import SpecDict, TextConfig
from programmatic_pid.validation import validate_spec


def load_spec(path: str | Path) -> SpecDict:
    """Load a YAML specification file.

    Precondition: *path* points to a valid YAML file containing a dict at root.
    Postcondition: returns a dict (possibly empty if YAML is blank).

    Raises:
        ValueError: if YAML root is not a dict (e.g., list, string, null).
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML spec in {path} must contain a dict at root, "
            f"not {type(data).__name__}"
        )
    return cast(SpecDict, data)


def prepare_spec(spec_path: str | Path, profile: str | None) -> SpecDict:
    """Load, validate, and apply profile to a spec.

    Postcondition: returned spec has passed validation twice (pre- and post-profile).
    """
    if spec_path is None:
        raise ValueError("spec_path must be provided")
    raw = load_spec(spec_path)
    validate_spec(raw)
    prepared = apply_profile(raw, profile)
    validate_spec(prepared)
    return prepared


class SpecAccessor:
    """Unified read-only access to spec configuration with defaults.

    DRY replacement for get_project, get_drawing, get_text_config,
    get_layout_config, get_layer_config.
    """

    def __init__(self, spec: SpecDict) -> None:
        self._spec = spec

    @property
    def spec(self) -> SpecDict:
        """Return the raw specification dictionary."""
        return self._spec

    @property
    def project(self) -> dict[str, Any]:
        """Return the top-level ``project`` section, or an empty dict if absent."""
        p = self._spec.get("project")
        return p if isinstance(p, dict) else {}

    @property
    def drawing(self) -> dict[str, Any]:
        """Return the drawing configuration dict.

        Checks ``spec["drawing"]`` first, then ``spec["project"]["drawing"]``.
        Returns an empty dict when neither is present.
        """
        d = self._spec.get("drawing")
        if isinstance(d, dict):
            return d
        proj_d = self.project.get("drawing")
        return proj_d if isinstance(proj_d, dict) else {}

    @property
    def text_config(self) -> TextConfig:
        drawing = self.drawing
        raw = drawing.get("text")
        if isinstance(raw, dict):
            return TextConfig(
                title_height=to_float(raw.get("title_height"), 3.2),
                subtitle_height=to_float(raw.get("subtitle_height"), 2.0),
                body_height=to_float(raw.get("body_height"), 1.6),
                small_height=to_float(raw.get("small_height"), 1.2),
            )
        base = to_float(drawing.get("text_height"), 2.5)
        if base <= 0:
            base = 2.5
        return TextConfig(
            title_height=base * 1.6,
            subtitle_height=base * 1.1,
            body_height=base,
            small_height=max(base * 0.75, 0.8),
        )

    @property
    def layer_config(self) -> dict[str, Any]:
        """Return the layer configuration dict.

        Checks ``drawing["layers"]`` then falls back to ``spec["layers"]``.
        Returns an empty dict when no layer config is defined.
        """
        drawing = self.drawing
        layers = drawing.get("layers")
        if isinstance(layers, dict) and layers:
            return layers
        layers = self._spec.get("layers")
        if isinstance(layers, dict):
            return layers
        return {}

    @property
    def layout_config(self) -> dict[str, Any]:
        """Return the layout configuration dict with validated defaults.

        All numeric values are clamped to safe minima so downstream
        rendering code never has to guard against nonsensical values.
        """
        drawing = self.drawing
        layout = drawing.get("layout", {})
        if not isinstance(layout, dict):
            layout = {}
        return {
            "style": str(layout.get("style", "clean")).lower(),
            "show_inline_equipment_notes": bool(
                layout.get("show_inline_equipment_notes", False)
            ),
            "show_instrument_suffix": bool(layout.get("show_instrument_suffix", False)),
            "show_control_tags_on_lines": bool(
                layout.get("show_control_tags_on_lines", False)
            ),
            "gap": max(to_float(layout.get("gap"), 8.0), 2.0),
            "right_panel_width": max(
                to_float(layout.get("right_panel_width"), 84.0), 45.0
            ),
            "bottom_panel_height": max(
                to_float(layout.get("bottom_panel_height"), 34.0), 18.0
            ),
            "title_block_height": max(
                to_float(layout.get("title_block_height"), 11.0), 6.0
            ),
            "panel_text_chars": max(int(layout.get("panel_text_chars", 42)), 24),
            "stream_label_scale": min(
                max(to_float(layout.get("stream_label_scale"), 0.76), 0.45), 1.5
            ),
            "stream_label_leaders": bool(layout.get("stream_label_leaders", True)),
            "instrument_spacing_factor": max(
                to_float(layout.get("instrument_spacing_factor"), 2.2), 1.2
            ),
            "controls_row_height_scale": max(
                to_float(layout.get("controls_row_height_scale"), 3.4), 2.0
            ),
        }

    @property
    def defaults(self) -> dict[str, Any]:
        """Return the ``defaults`` section, or an empty dict if absent."""
        d = self._spec.get("defaults", {})
        return d if isinstance(d, dict) else {}

    @property
    def equipment(self) -> list[dict[str, Any]]:
        """Return the list of equipment entries, or ``[]`` if absent."""
        v = self._spec.get("equipment")
        return cast(list[dict[str, Any]], v) if isinstance(v, list) else []

    @property
    def instruments(self) -> list[dict[str, Any]]:
        """Return the list of instrument entries, or ``[]`` if absent."""
        v = self._spec.get("instruments")
        return cast(list[dict[str, Any]], v) if isinstance(v, list) else []

    @property
    def streams(self) -> list[dict[str, Any]]:
        """Return the list of process stream entries, or ``[]`` if absent."""
        v = self._spec.get("streams")
        return cast(list[dict[str, Any]], v) if isinstance(v, list) else []

    @property
    def control_loops(self) -> list[dict[str, Any]]:
        """Return the list of control loop entries, or ``[]`` if absent."""
        v = self._spec.get("control_loops")
        return cast(list[dict[str, Any]], v) if isinstance(v, list) else []

    @property
    def interlocks(self) -> list[dict[str, Any]]:
        """Return the list of interlock entries, or ``[]`` if absent."""
        v = self._spec.get("interlocks")
        return cast(list[dict[str, Any]], v) if isinstance(v, list) else []


# ---------------------------------------------------------------------------
# Backward-compatible free functions that delegate to the old interface.
# These are kept so that generator.py continues to work during migration.
# ---------------------------------------------------------------------------
def get_project(spec: SpecDict) -> dict[str, Any]:
    """Return the ``project`` section from *spec*, or an empty dict.

    Deprecated: use ``SpecAccessor(spec).project`` instead.
    """
    p = spec.get("project")
    return p if isinstance(p, dict) else {}


def get_drawing(spec: SpecDict) -> dict[str, Any]:
    """Return the drawing configuration dict from *spec*.

    Checks ``spec["drawing"]`` first, then ``spec["project"]["drawing"]``.

    Deprecated: use ``SpecAccessor(spec).drawing`` instead.
    """
    if "drawing" in spec and isinstance(spec["drawing"], dict):
        return spec["drawing"]
    d = get_project(spec).get("drawing")
    return cast(dict[str, Any], d) if isinstance(d, dict) else {}


def ensure_drawing(spec: SpecDict) -> dict[str, Any]:
    """Return or create the drawing dict inside *spec* (mutates *spec*).

    Ensures that ``spec["project"]["drawing"]`` exists and returns it.
    Used during spec preparation before rendering.

    Deprecated: use ``SpecAccessor`` for read-only access instead.
    """
    if "drawing" in spec and isinstance(spec["drawing"], dict):
        return spec["drawing"]
    project = spec.setdefault("project", {})
    drawing = project.get("drawing")
    if not isinstance(drawing, dict):
        drawing = {}
        project["drawing"] = drawing
    return cast(dict[str, Any], drawing)


def get_text_config(spec: SpecDict) -> dict[str, float]:
    """Return text height configuration as a plain dict.

    Returns keys: ``title_height``, ``subtitle_height``, ``body_height``,
    ``small_height`` (all floats in drawing units).

    Deprecated: use ``SpecAccessor(spec).text_config`` instead.
    """
    tc = SpecAccessor(spec).text_config
    return {
        "title_height": tc.title_height,
        "subtitle_height": tc.subtitle_height,
        "body_height": tc.body_height,
        "small_height": tc.small_height,
    }


def get_layout_config(spec: SpecDict) -> dict[str, Any]:
    """Return the layout configuration dict with validated defaults.

    Deprecated: use ``SpecAccessor(spec).layout_config`` instead.
    """
    return SpecAccessor(spec).layout_config


def get_layer_config(spec: SpecDict) -> dict[str, Any]:
    """Return the layer configuration dict.

    Deprecated: use ``SpecAccessor(spec).layer_config`` instead.
    """
    return SpecAccessor(spec).layer_config
