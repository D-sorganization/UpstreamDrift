"""Unit tests for :class:`PlotStyleSpec` and :class:`PlotStyleSet` persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.plot_style import (
    SCHEMA_VERSION,
    ColormapId,
    DataChannel,
    DataDrivenColor,
    MarkerShape,
    MarkerStyle,
    PaletteColor,
    PlotStyleSet,
    PlotStyleSpec,
    StaticColor,
)


# ---------- PlotStyleSpec ----------------------------------------------


def test_plot_style_spec_happy_path() -> None:
    spec = PlotStyleSpec(
        name="body",
        target="marker_group:body",
        style=MarkerStyle(),
    )
    assert spec.name == "body"


def test_plot_style_spec_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PlotStyleSpec(name="", target="t", style=MarkerStyle())


def test_plot_style_spec_rejects_empty_target() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        PlotStyleSpec(name="x", target="", style=MarkerStyle())


def test_plot_style_spec_rejects_non_marker_style() -> None:
    with pytest.raises(TypeError, match="MarkerStyle"):
        PlotStyleSpec(name="x", target="t", style="not a style")  # type: ignore[arg-type]


# ---------- PlotStyleSet -----------------------------------------------


def test_plot_style_set_default_construction() -> None:
    style_set = PlotStyleSet()
    assert style_set.schema_version == SCHEMA_VERSION
    assert style_set.entries == ()


def test_plot_style_set_rejects_duplicate_names() -> None:
    a = PlotStyleSpec(name="x", target="t1", style=MarkerStyle())
    b = PlotStyleSpec(name="x", target="t2", style=MarkerStyle())
    with pytest.raises(ValueError, match="duplicate"):
        PlotStyleSet(entries=(a, b))


def test_plot_style_set_rejects_non_int_schema_version() -> None:
    with pytest.raises(TypeError, match="schema_version"):
        PlotStyleSet(schema_version="1")  # type: ignore[arg-type]


def test_plot_style_set_rejects_zero_schema_version() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        PlotStyleSet(schema_version=0)


def test_plot_style_set_rejects_non_tuple_entries() -> None:
    with pytest.raises(TypeError, match="tuple"):
        PlotStyleSet(entries=[])  # type: ignore[arg-type]


def test_plot_style_set_rejects_non_spec_entry() -> None:
    with pytest.raises(TypeError, match="PlotStyleSpec"):
        PlotStyleSet(entries=("not a spec",))  # type: ignore[arg-type]


# ---------- JSON round-trip --------------------------------------------


def _make_set_with_static_only() -> PlotStyleSet:
    spec = PlotStyleSpec(
        name="body",
        target="marker_group:body",
        style=MarkerStyle(
            shape=MarkerShape.SPHERE,
            size_px=8.0,
            edge_color="#000000",
            edge_width=0.5,
            fill_color=StaticColor("#1f77b4"),
            opacity=0.9,
        ),
    )
    return PlotStyleSet(entries=(spec,))


def _make_set_with_palette() -> PlotStyleSet:
    spec = PlotStyleSpec(
        name="club",
        target="trace:clubhead",
        style=MarkerStyle(
            shape=MarkerShape.STAR,
            fill_color=PaletteColor(palette_name="tab10", palette_index=3),
        ),
    )
    return PlotStyleSet(entries=(spec,))


def _make_set_with_data_driven() -> tuple[PlotStyleSet, DataChannel]:
    channel = DataChannel.from_array("speed", np.array([0.0, 5.0, 10.0]), unit="m/s")
    spec = PlotStyleSpec(
        name="speed",
        target="trace:hand",
        style=MarkerStyle(
            shape=MarkerShape.DIAMOND,
            fill_color=DataDrivenColor(
                channel=channel,
                colormap=ColormapId.VELOCITY,
                vmin=0.0,
                vmax=10.0,
                nan_color="#444444",
            ),
        ),
    )
    return PlotStyleSet(entries=(spec,)), channel


def test_round_trip_static_color(tmp_path: Path) -> None:
    style_set = _make_set_with_static_only()
    path = tmp_path / "set.json"
    style_set.save(path)
    loaded = PlotStyleSet.load(path)
    assert loaded == style_set


def test_round_trip_palette_color(tmp_path: Path) -> None:
    style_set = _make_set_with_palette()
    path = tmp_path / "set.json"
    style_set.save(path)
    loaded = PlotStyleSet.load(path)
    assert loaded == style_set


def test_round_trip_data_driven_color(tmp_path: Path) -> None:
    style_set, channel = _make_set_with_data_driven()
    path = tmp_path / "set.json"
    style_set.save(path)
    loaded = PlotStyleSet.load(path, channel_lookup={channel.name: channel})
    # The reconstructed channel matches by name; equality of arrays is
    # covered by reusing the supplied lookup.
    assert loaded == style_set


def test_round_trip_data_driven_without_lookup(tmp_path: Path) -> None:
    """Without channel_lookup, the loader builds a placeholder channel."""
    style_set, _ = _make_set_with_data_driven()
    path = tmp_path / "set.json"
    style_set.save(path)
    loaded = PlotStyleSet.load(path)
    # Same colormap, same name; placeholder channel has zero-length values.
    assert len(loaded.entries) == 1
    fill = loaded.entries[0].style.fill_color
    assert isinstance(fill, DataDrivenColor)
    assert fill.channel.name == "speed"
    assert fill.channel.values.size == 0


def test_unknown_top_level_keys_are_tolerated() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": [],
        "future_field": "ignored",
        "another_unknown": {"nested": True},
    }
    style_set = PlotStyleSet.from_json(payload)
    assert style_set.entries == ()


def test_unknown_entry_keys_are_tolerated() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": [
            {
                "name": "body",
                "target": "marker_group:body",
                "style": {
                    "shape": "sphere",
                    "size_px": 6.0,
                    "edge_color": "#000000",
                    "edge_width": 0.5,
                    "fill_color": {"kind": "static", "hex_value": "#1f77b4"},
                    "opacity": 1.0,
                    "future_field": "ignored",
                },
                "label": "ignored",
            }
        ],
    }
    style_set = PlotStyleSet.from_json(payload)
    assert len(style_set.entries) == 1


def test_load_reads_actual_json_file(tmp_path: Path) -> None:
    path = tmp_path / "set.json"
    payload = {"schema_version": SCHEMA_VERSION, "entries": []}
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = PlotStyleSet.load(path)
    assert loaded == PlotStyleSet()


def test_save_writes_pretty_json(tmp_path: Path) -> None:
    style_set = _make_set_with_static_only()
    path = tmp_path / "set.json"
    style_set.save(path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "  " in text  # indented


def test_save_rejects_custom_mesh(tmp_path: Path) -> None:
    """CUSTOM_MESH is not v1-serialisable."""
    import numpy as np

    from src.shared.python.plot_style import CustomMeshSpec

    mesh = CustomMeshSpec(
        name="t",
        vertices=np.zeros((3, 3)),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    spec = PlotStyleSpec(
        name="m",
        target="t",
        style=MarkerStyle(shape=MarkerShape.CUSTOM_MESH, custom_mesh=mesh),
    )
    style_set = PlotStyleSet(entries=(spec,))
    with pytest.raises(ValueError, match="CUSTOM_MESH"):
        style_set.save(tmp_path / "set.json")


def test_save_rejects_non_path() -> None:
    style_set = PlotStyleSet()
    with pytest.raises(TypeError, match="pathlib.Path"):
        style_set.save("not_a_path")  # type: ignore[arg-type]


def test_load_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        PlotStyleSet.load("not_a_path")  # type: ignore[arg-type]


def test_from_json_rejects_non_dict() -> None:
    with pytest.raises(TypeError, match="dict"):
        PlotStyleSet.from_json("not a dict")  # type: ignore[arg-type]


def test_from_json_rejects_non_list_entries() -> None:
    with pytest.raises(TypeError, match="list"):
        PlotStyleSet.from_json({"entries": "nope"})


def test_from_json_unknown_color_kind_raises() -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": [
            {
                "name": "x",
                "target": "t",
                "style": {
                    "shape": "sphere",
                    "fill_color": {"kind": "mystery"},
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown"):
        PlotStyleSet.from_json(payload)
