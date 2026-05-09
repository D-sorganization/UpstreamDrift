"""Tests for ``body_part_viz.theme``.

Covers :class:`ShapeTheme` validation: color string forms, opacity and
edge-width ranges, frozen-dataclass invariant, and group string non-emptiness.
"""

from __future__ import annotations

import pytest

from src.shared.python.body_part_viz.theme import ShapeTheme


# ---------------------------------------------------------------------------
# Defaults + happy paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_defaults() -> None:
    t = ShapeTheme()
    assert t.color == "#1f77b4"
    assert t.opacity == 0.8
    assert t.edge_color == "#000000"
    assert t.edge_width == 0.5
    assert t.flat_shaded is True
    assert t.group == "default"


@pytest.mark.unit
@pytest.mark.parametrize(
    "color",
    [
        "#fff",
        "#ffff",
        "#abcdef",
        "#1f77b4",
        "#1f77b4ff",
        "red",
        "tab:blue",
        "C0",
        "darkslategray",
    ],
)
def test_accepts_valid_color_forms(color: str) -> None:
    ShapeTheme(color=color, edge_color=color)


@pytest.mark.unit
def test_accepts_zero_opacity() -> None:
    ShapeTheme(opacity=0.0)


@pytest.mark.unit
def test_accepts_full_opacity() -> None:
    ShapeTheme(opacity=1.0)


@pytest.mark.unit
def test_accepts_zero_edge_width() -> None:
    ShapeTheme(edge_width=0.0)


# ---------------------------------------------------------------------------
# Frozen
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_theme_is_frozen() -> None:
    t = ShapeTheme()
    with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
        t.color = "#ffffff"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Color validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_color_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="color must be non-empty"):
        ShapeTheme(color="")


@pytest.mark.unit
def test_color_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="color must be str"):
        ShapeTheme(color=42)  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_hex",
    [
        "#ff",  # too short
        "#fffff",  # 5 chars
        "#fffffff",  # 7 chars
        "#zzzzzz",  # invalid hex digits
        "#1f77b4ffff",  # too long
    ],
)
def test_color_rejects_invalid_hex(bad_hex: str) -> None:
    with pytest.raises(ValueError, match="not a valid hex color"):
        ShapeTheme(color=bad_hex)


@pytest.mark.unit
def test_color_rejects_starts_with_digit() -> None:
    """Named colors must start with an alpha character."""
    with pytest.raises(ValueError, match="not a valid color"):
        ShapeTheme(color="123red")


@pytest.mark.unit
def test_edge_color_rejects_invalid_hex() -> None:
    with pytest.raises(ValueError, match="edge_color"):
        ShapeTheme(edge_color="#zz")


# ---------------------------------------------------------------------------
# Opacity validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_opacity_rejects_above_one() -> None:
    with pytest.raises(ValueError, match=r"opacity must be in \[0.0, 1.0\]"):
        ShapeTheme(opacity=1.1)


@pytest.mark.unit
def test_opacity_rejects_below_zero() -> None:
    with pytest.raises(ValueError, match=r"opacity must be in \[0.0, 1.0\]"):
        ShapeTheme(opacity=-0.1)


@pytest.mark.unit
def test_opacity_rejects_inf() -> None:
    with pytest.raises(ValueError, match="opacity must be finite"):
        ShapeTheme(opacity=float("inf"))


@pytest.mark.unit
def test_opacity_rejects_nan() -> None:
    with pytest.raises(ValueError, match="opacity must be finite"):
        ShapeTheme(opacity=float("nan"))


@pytest.mark.unit
def test_opacity_rejects_string() -> None:
    with pytest.raises(TypeError, match="opacity must be float"):
        ShapeTheme(opacity="0.5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Edge width validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_edge_width_rejects_negative() -> None:
    with pytest.raises(ValueError, match="edge_width must be >= 0"):
        ShapeTheme(edge_width=-0.5)


@pytest.mark.unit
def test_edge_width_rejects_inf() -> None:
    with pytest.raises(ValueError, match="edge_width must be finite"):
        ShapeTheme(edge_width=float("inf"))


@pytest.mark.unit
def test_edge_width_rejects_string() -> None:
    with pytest.raises(TypeError, match="edge_width must be float"):
        ShapeTheme(edge_width="0.5")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Other field validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flat_shaded_rejects_int() -> None:
    """Strict type: flat_shaded must be a bool, not int (per docstring)."""
    with pytest.raises(TypeError, match="flat_shaded must be bool"):
        ShapeTheme(flat_shaded=1)  # type: ignore[arg-type]


@pytest.mark.unit
def test_group_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="group must be non-empty"):
        ShapeTheme(group="")


@pytest.mark.unit
def test_group_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="group must be str"):
        ShapeTheme(group=42)  # type: ignore[arg-type]
