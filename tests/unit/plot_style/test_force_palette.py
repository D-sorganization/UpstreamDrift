"""Tests for FORCE_KIND_PALETTE in plot_style (FTO-3, #11288)."""

from __future__ import annotations

import pytest

from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.plot_style.force_palette import (
    FORCE_KIND_PALETTE,
    get_kind_rgba,
    hex_to_rgba,
)

pytestmark = pytest.mark.unit


def test_force_kind_palette_contains_all_wrench_kinds() -> None:
    expected_kinds = {
        "joint_actuator",
        "joint_reaction",
        "contact",
        "grip",
        "external",
        "gravity",
        "muscle",
    }
    assert set(FORCE_KIND_PALETTE.keys()) == expected_kinds
    for kind in WrenchKind:
        assert kind.value in FORCE_KIND_PALETTE
        assert kind in FORCE_KIND_PALETTE


def test_force_kind_palette_expected_hex_values() -> None:
    expected = {
        "joint_actuator": "#E69F00",
        "joint_reaction": "#CC79A7",
        "contact": "#009E73",
        "grip": "#56B4E9",
        "external": "#000000",
        "gravity": "#999999",
        "muscle": "#D55E00",
    }
    for k, hex_val in expected.items():
        assert FORCE_KIND_PALETTE[k].upper() == hex_val.upper()


def test_force_kind_palette_reserves_tension_and_compression_colors() -> None:
    reserved = {"#0000ff", "#ff0000"}
    for hex_val in FORCE_KIND_PALETTE.values():
        assert hex_val.lower() not in reserved


def test_hex_to_rgba_valid() -> None:
    assert hex_to_rgba("#ff0000") == (1.0, 0.0, 0.0, 1.0)
    assert hex_to_rgba("#00ff00", alpha=0.5) == (0.0, 1.0, 0.0, 0.5)
    assert hex_to_rgba("#0000ff80") == pytest.approx(
        (0.0, 0.0, 1.0, 128 / 255.0), abs=1e-3
    )
    assert hex_to_rgba("#fff") == (1.0, 1.0, 1.0, 1.0)


def test_hex_to_rgba_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid hex color"):
        hex_to_rgba("not_a_hex")
    with pytest.raises(ValueError, match="Invalid hex color"):
        hex_to_rgba("#12345")


def test_get_kind_rgba() -> None:
    rgba = get_kind_rgba(WrenchKind.CONTACT)
    assert len(rgba) == 4
    assert rgba[3] == 1.0
    for c in rgba:
        assert 0.0 <= c <= 1.0
