"""Unit tests for plot_style color utilities and tension/compression colormap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.body_part_viz import ForceColorScale
from src.shared.python.plot_style import ColormapId, get_colormap
from src.shared.python.plot_style.color_utils import hex_to_rgba, rgba_to_hex


# ============================================================================
# hex_to_rgba tests
# ============================================================================


def test_hex_to_rgba_standard_6_digit() -> None:
    assert hex_to_rgba("#0000ff") == (0.0, 0.0, 1.0, 1.0)
    assert hex_to_rgba("#ff0000") == (1.0, 0.0, 0.0, 1.0)
    assert hex_to_rgba("#ffffff") == (1.0, 1.0, 1.0, 1.0)
    assert hex_to_rgba("#000000") == (0.0, 0.0, 0.0, 1.0)


def test_hex_to_rgba_custom_alpha() -> None:
    assert hex_to_rgba("#0000ff", alpha=0.5) == (0.0, 0.0, 1.0, 0.5)
    assert hex_to_rgba("#ff0000", alpha=0.0) == (1.0, 0.0, 0.0, 0.0)


def test_hex_to_rgba_short_3_digit_expansion() -> None:
    expected_abc = (0xAA / 255.0, 0xBB / 255.0, 0xCC / 255.0, 1.0)
    assert hex_to_rgba("#abc") == pytest.approx(expected_abc)
    assert hex_to_rgba("#ABC") == pytest.approx(expected_abc)
    assert hex_to_rgba("#f0a") == pytest.approx((1.0, 0.0, 0xAA / 255.0, 1.0))


def test_hex_to_rgba_8_digit_with_alpha() -> None:
    assert hex_to_rgba("#0000ff80") == pytest.approx((0.0, 0.0, 1.0, 0x80 / 255.0))
    assert hex_to_rgba("#11223344") == pytest.approx(
        (0x11 / 255.0, 0x22 / 255.0, 0x33 / 255.0, 0x44 / 255.0)
    )


def test_hex_to_rgba_invalid_strings_raise_value_error() -> None:
    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("blue")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("#12")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("#")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("#1234")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("#12345")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("#1234567")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("#123456789")

    with pytest.raises(ValueError, match="hex"):
        hex_to_rgba("#gggggg")


def test_hex_to_rgba_invalid_types_and_alpha() -> None:
    with pytest.raises(TypeError):
        hex_to_rgba(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        hex_to_rgba("#0000ff", alpha="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        hex_to_rgba("#0000ff", alpha=-0.1)

    with pytest.raises(ValueError):
        hex_to_rgba("#0000ff", alpha=1.1)

    with pytest.raises(ValueError):
        hex_to_rgba("#0000ff", alpha=float("nan"))


# ============================================================================
# rgba_to_hex tests
# ============================================================================


def test_rgba_to_hex_4_tuple() -> None:
    assert rgba_to_hex((0.0, 0.0, 1.0, 1.0)) == "#0000ffff"
    assert rgba_to_hex((0.0, 0.0, 1.0, 1.0), include_alpha=False) == "#0000ff"
    assert rgba_to_hex((1.0, 0.0, 0.0, 0.5)) == "#ff000080"
    assert rgba_to_hex((1.0, 1.0, 1.0, 0.0)) == "#ffffff00"


def test_rgba_to_hex_3_tuple() -> None:
    assert rgba_to_hex((0.0, 0.0, 1.0)) == "#0000ff"
    assert rgba_to_hex((1.0, 0.0, 0.0), include_alpha=False) == "#ff0000"


def test_rgba_to_hex_validation() -> None:
    # Component bounds
    with pytest.raises(ValueError, match="[0.0, 1.0]"):
        rgba_to_hex((-0.01, 0.0, 1.0))

    with pytest.raises(ValueError, match="[0.0, 1.0]"):
        rgba_to_hex((1.01, 0.0, 1.0))

    with pytest.raises(ValueError, match="finite"):
        rgba_to_hex((float("nan"), 0.0, 1.0))

    with pytest.raises(ValueError, match="finite"):
        rgba_to_hex((0.0, float("inf"), 1.0))

    # Length
    with pytest.raises(ValueError, match="components"):
        rgba_to_hex((0.0, 1.0))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="components"):
        rgba_to_hex((0.0, 1.0, 0.0, 1.0, 0.5))  # type: ignore[arg-type]

    # Type
    with pytest.raises(TypeError):
        rgba_to_hex("not-a-tuple")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        rgba_to_hex((0.0, "red", 1.0))  # type: ignore[arg-type]


# ============================================================================
# Round trip tests
# ============================================================================


@pytest.mark.parametrize(
    "hex_sample",
    [
        "#00000000",
        "#ffffffff",
        "#11223344",
        "#aabbccdd",
        "#12345678",
        "#0000ff80",
    ],
)
def test_round_trip_8_digit(hex_sample: str) -> None:
    assert rgba_to_hex(hex_to_rgba(hex_sample)) == hex_sample.lower()


@pytest.mark.parametrize(
    "hex_sample",
    [
        "#000000",
        "#ffffff",
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#123456",
        "#abcdef",
    ],
)
def test_round_trip_6_digit_without_alpha(hex_sample: str) -> None:
    assert (
        rgba_to_hex(hex_to_rgba(hex_sample), include_alpha=False) == hex_sample.lower()
    )


# ============================================================================
# ColormapId.TENSION_COMPRESSION tests
# ============================================================================


def test_tension_compression_colormap_registered_and_evaluated() -> None:
    cmap = get_colormap(ColormapId.TENSION_COMPRESSION)
    assert cmap is not None

    # Check evaluated colors at 0.0, 0.5, 1.0
    scale = ForceColorScale()
    expected_compression = hex_to_rgba(scale.compression_color)
    expected_neutral = hex_to_rgba(scale.neutral_color)
    expected_tension = hex_to_rgba(scale.tension_color)

    np.testing.assert_allclose(cmap(0.0), expected_compression, atol=1e-3)
    np.testing.assert_allclose(cmap(0.5), expected_neutral, atol=1e-3)
    np.testing.assert_allclose(cmap(1.0), expected_tension, atol=1e-3)

    # By default, ForceColorScale is compression=#ff0000, neutral=#ffffff, tension=#0000ff
    np.testing.assert_allclose(cmap(0.0), (1.0, 0.0, 0.0, 1.0), atol=1e-3)
    np.testing.assert_allclose(cmap(0.5), (1.0, 1.0, 1.0, 1.0), atol=1e-3)
    np.testing.assert_allclose(cmap(1.0), (0.0, 0.0, 1.0, 1.0), atol=1e-3)


def test_tension_compression_string_lookup() -> None:
    cmap1 = get_colormap(ColormapId.TENSION_COMPRESSION)
    cmap2 = get_colormap("tension_compression")
    assert cmap1.name == cmap2.name
