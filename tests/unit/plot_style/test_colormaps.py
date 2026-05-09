"""Unit tests for :class:`ColormapId` and :class:`CustomColormap`."""

from __future__ import annotations

import pytest

from src.shared.python.plot_style import ColormapId, CustomColormap
from src.shared.python.plot_style.colormaps import (
    SEMANTIC_COLORMAP_ALIASES,
    resolve_colormap_alias,
)


# ---------- ColormapId --------------------------------------------------


def test_colormap_id_round_trip_through_string() -> None:
    for cmap_id in ColormapId:
        assert str(cmap_id) == cmap_id.value
        assert ColormapId(cmap_id.value) is cmap_id


def test_resolve_alias_passes_through_builtins() -> None:
    assert resolve_colormap_alias(ColormapId.VIRIDIS) is ColormapId.VIRIDIS


def test_resolve_alias_resolves_semantic_to_builtin() -> None:
    for alias, target in SEMANTIC_COLORMAP_ALIASES.items():
        assert resolve_colormap_alias(alias) is target


def test_resolve_alias_rejects_non_enum() -> None:
    with pytest.raises(TypeError, match="ColormapId"):
        resolve_colormap_alias("viridis")  # type: ignore[arg-type]


# ---------- CustomColormap ---------------------------------------------


def test_custom_colormap_happy_path() -> None:
    cmap = CustomColormap(
        name="my_cmap",
        stops=((0.0, "#000000"), (0.5, "#888888"), (1.0, "#ffffff")),
    )
    assert cmap.name == "my_cmap"
    assert len(cmap.stops) == 3


def test_custom_colormap_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CustomColormap(name="", stops=((0.0, "#000"), (1.0, "#fff")))


def test_custom_colormap_rejects_non_tuple_stops() -> None:
    with pytest.raises(TypeError, match="tuple"):
        CustomColormap(
            name="x",
            stops=[(0.0, "#000"), (1.0, "#fff")],  # type: ignore[arg-type]
        )


def test_custom_colormap_rejects_too_few_stops() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        CustomColormap(name="x", stops=((0.0, "#000"),))


def test_custom_colormap_rejects_non_monotonic_stops() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CustomColormap(
            name="x",
            stops=((0.0, "#000"), (0.5, "#888"), (0.4, "#fff")),
        )


def test_custom_colormap_rejects_duplicate_positions() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CustomColormap(
            name="x",
            stops=((0.0, "#000"), (0.5, "#888"), (0.5, "#fff")),
        )


def test_custom_colormap_rejects_position_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CustomColormap(name="x", stops=((-0.1, "#000"), (1.0, "#fff")))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CustomColormap(name="x", stops=((0.0, "#000"), (1.5, "#fff")))


def test_custom_colormap_rejects_non_parseable_hex() -> None:
    with pytest.raises(ValueError, match="parseable"):
        CustomColormap(name="x", stops=((0.0, "not_a_color"), (1.0, "#fff")))


def test_custom_colormap_rejects_malformed_stop_tuple() -> None:
    with pytest.raises(ValueError, match="2-tuple"):
        CustomColormap(name="x", stops=((0.0,), (1.0, "#fff")))  # type: ignore[arg-type]


def test_custom_colormap_rejects_non_numeric_position() -> None:
    with pytest.raises(TypeError, match="numeric"):
        CustomColormap(
            name="x",
            stops=(("zero", "#000"), (1.0, "#fff")),  # type: ignore[arg-type]
        )


def test_custom_colormap_rejects_empty_hex() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CustomColormap(name="x", stops=((0.0, ""), (1.0, "#fff")))
