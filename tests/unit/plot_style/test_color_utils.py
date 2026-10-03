"""Unit tests for plot_style.color_utils (FTO-4, #11289)."""

from __future__ import annotations

import pytest

from src.shared.python.plot_style.color_utils import hex_to_rgba, rgba_to_hex

pytestmark = pytest.mark.unit


def test_hex_to_rgba_valid() -> None:
    """hex_to_rgba converts #rgb, #rrggbb and #rrggbbaa strings to float tuples in [0, 1]."""
    assert hex_to_rgba("#0000ff") == (0.0, 0.0, 1.0, 1.0)
    assert hex_to_rgba("#ff0000") == (1.0, 0.0, 0.0, 1.0)
    assert hex_to_rgba("#ffffff") == (1.0, 1.0, 1.0, 1.0)
    assert hex_to_rgba("#000000") == (0.0, 0.0, 0.0, 1.0)

    # #abc expands to #aabbcc
    r, g, b, a = hex_to_rgba("#abc")
    assert r == pytest.approx(170 / 255.0)
    assert g == pytest.approx(187 / 255.0)
    assert b == pytest.approx(204 / 255.0)
    assert a == 1.0

    # Custom alpha override for 6-digit hex
    assert hex_to_rgba("#0000ff", alpha=0.5) == (0.0, 0.0, 1.0, 0.5)

    # 8-digit hex (#rrggbbaa)
    r8, g8, b8, a8 = hex_to_rgba("#0000ff80")
    assert r8 == 0.0
    assert g8 == 0.0
    assert b8 == 1.0
    assert a8 == pytest.approx(128 / 255.0)

    # 8-digit hex with custom alpha scaling
    r8_sc, g8_sc, b8_sc, a8_sc = hex_to_rgba("#0000ff80", alpha=0.5)
    assert a8_sc == pytest.approx((128 / 255.0) * 0.5)


def test_hex_to_rgba_invalid_raises() -> None:
    """Invalid hex color strings and invalid alpha values raise ValueError or TypeError."""
    # Named colors must raise
    with pytest.raises(ValueError, match="Invalid hex color string"):
        hex_to_rgba("blue")

    # Incomplete or invalid length hex strings
    for invalid in (
        "",
        "#",
        "#1",
        "#12",
        "#1234",
        "#12345",
        "#1234567",
        "#123456789",
        "0000ff",
    ):
        with pytest.raises(ValueError, match="Invalid hex color string"):
            hex_to_rgba(invalid)

    # Invalid hex characters
    with pytest.raises(ValueError, match="Invalid hex color string"):
        hex_to_rgba("#xyz")
    with pytest.raises(ValueError, match="Invalid hex color string"):
        hex_to_rgba("#12345g")

    # Type validation
    with pytest.raises(TypeError, match="hex color must be str"):
        hex_to_rgba(123456)  # type: ignore[arg-type]

    # Alpha validation
    with pytest.raises(ValueError, match="alpha must be in"):
        hex_to_rgba("#0000ff", alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        hex_to_rgba("#0000ff", alpha=1.1)
    with pytest.raises(TypeError, match="alpha must be a real number"):
        hex_to_rgba("#0000ff", alpha=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alpha must be finite"):
        hex_to_rgba("#0000ff", alpha=float("nan"))


def test_rgba_to_hex_round_trip() -> None:
    """rgba_to_hex(hex_to_rgba(x)) == x.lower() round trips for sample hex values."""
    samples = [
        "#000000",
        "#ffffff",
        "#ff0000",
        "#00ff00",
        "#0000ff",
        "#123456",
        "#abcdef",
        "#aabbcc",
        "#336699",
    ]
    for sample in samples:
        assert rgba_to_hex(hex_to_rgba(sample)) == sample.lower()

    # Alpha samples
    alpha_samples = [
        "#11223380",
        "#ff000020",
        "#0000ffaa",
    ]
    for sample in alpha_samples:
        assert rgba_to_hex(hex_to_rgba(sample)) == sample.lower()


def test_rgba_to_hex_options_and_validation() -> None:
    """rgba_to_hex flags and input validation adhere to DbC."""
    # 3-element tuple
    assert rgba_to_hex((1.0, 0.0, 0.0)) == "#ff0000"

    # include_alpha flag
    assert rgba_to_hex((0.0, 0.0, 1.0, 1.0), include_alpha=True) == "#0000ffff"
    assert rgba_to_hex((0.0, 0.0, 1.0, 0.5), include_alpha=False) == "#0000ff"

    # Invalid lengths
    with pytest.raises(ValueError, match="rgba must have length 3 or 4"):
        rgba_to_hex((1.0, 0.0))
    with pytest.raises(ValueError, match="rgba must have length 3 or 4"):
        rgba_to_hex((1.0, 0.0, 0.0, 1.0, 0.5))

    # Out of range / non-finite
    with pytest.raises(ValueError, match="component at index 0 must be in"):
        rgba_to_hex((-0.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="component at index 0 must be in"):
        rgba_to_hex((1.1, 0.0, 0.0))
    with pytest.raises(ValueError, match="component at index 0 must be finite"):
        rgba_to_hex((float("nan"), 0.0, 0.0))

    # Non-numeric
    with pytest.raises(TypeError, match="component at index 0 must be a real number"):
        rgba_to_hex((True, 0.0, 0.0))  # type: ignore[arg-type]
