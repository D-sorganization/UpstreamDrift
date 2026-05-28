"""TDD coverage for the new pure colour-math helpers.

These three helpers landed today in ``src/shared/python/theme/color_derivation.py``
to back ``ThemeColors.model_post_init``. They must:

* never raise on malformed input (a typo'd hex must not crash launcher boot),
* always return ``#rrggbb`` or ``#rrggbbaa`` on success,
* classify dark / light backgrounds via the standard 0.299/0.587/0.114
  luminance rule.
"""

from __future__ import annotations

import pytest

from src.shared.python.theme.color_derivation import (
    adjust,
    is_dark_bg,
    with_alpha,
)


# ── is_dark_bg ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bg,expected",
    [
        ("#000000", True),
        ("#ffffff", False),
        ("#1a1d23", True),  # default dark bg
        ("#f8f9fa", False),  # default light bg
        ("#fff", False),  # 3-digit shorthand
        ("#000", True),
    ],
)
def test_is_dark_bg_classifies_correctly(bg: str, expected: bool) -> None:
    assert is_dark_bg(bg) is expected


@pytest.mark.parametrize(
    "bad",
    ["", "not-a-color", "#zzzzzz", "#12", "#12345"],
)
def test_is_dark_bg_tolerates_garbage(bad: str) -> None:
    """DbC: ``is_dark_bg`` must never raise, only return a bool."""
    assert isinstance(is_dark_bg(bad), bool)


# ── adjust ──────────────────────────────────────────────────────────────


def test_adjust_lighter_factor_above_one() -> None:
    out = adjust("#808080", 1.5)
    # 0x80 * 1.5 == 192 == 0xc0
    assert out.lower() == "#c0c0c0"


def test_adjust_darker_factor_below_one() -> None:
    out = adjust("#808080", 0.5)
    assert out.lower() == "#404040"


def test_adjust_clamps_to_255() -> None:
    out = adjust("#ffffff", 2.0)
    assert out.lower() == "#ffffff"


def test_adjust_clamps_to_zero() -> None:
    out = adjust("#ffffff", 0.0)
    assert out.lower() == "#000000"


@pytest.mark.parametrize("bad", ["", "not-hex", "#12", "#zzz"])
def test_adjust_passes_garbage_through(bad: str) -> None:
    """Robustness: invalid colour returns input unchanged, never raises."""
    assert adjust(bad, 1.2) == bad


def test_adjust_expands_three_digit_shorthand() -> None:
    assert adjust("#f00", 1.0).lower() == "#ff0000"


# ── with_alpha ──────────────────────────────────────────────────────────


def test_with_alpha_appends_alpha_byte() -> None:
    assert with_alpha("#ff0000", 0x80).lower() == "#ff000080"


def test_with_alpha_clamps_high() -> None:
    assert with_alpha("#ff0000", 9999).lower() == "#ff0000ff"


def test_with_alpha_clamps_negative() -> None:
    assert with_alpha("#ff0000", -1).lower() == "#ff000000"


def test_with_alpha_handles_shorthand() -> None:
    assert with_alpha("#f00", 0x40).lower() == "#ff000040"


@pytest.mark.parametrize("bad", ["", "#12", "#1"])
def test_with_alpha_passes_short_input_through(bad: str) -> None:
    """Inputs shorter than 6 hex chars are returned unchanged.

    Longer non-hex inputs aren't validated as hex by the helper — the
    schema-derivation pipeline never feeds it arbitrary strings, only
    output of ``adjust``/``_expand`` which are already #rrggbb shaped.
    """
    assert with_alpha(bad, 0x80) == bad
