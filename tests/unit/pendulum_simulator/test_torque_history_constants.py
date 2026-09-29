"""Tests for src.shared.python.pendulum_simulator.gui.torque_history_constants (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.pendulum_simulator.gui.torque_history_constants import (
    get_drive_colors,
    get_friction_colors,
    get_total_colors,
    set_colorblind_mode,
)


def setup_function() -> None:
    """Reset to default palette before each test."""
    set_colorblind_mode(False)


def teardown_function() -> None:
    """Reset after each test."""
    set_colorblind_mode(False)


class TestGetDriveColors:
    def test_torque_history_constants_returns_list(self) -> None:
        colors = get_drive_colors()
        assert isinstance(colors, list)

    def test_torque_history_constants_non_empty(self) -> None:
        colors = get_drive_colors()
        assert len(colors) > 0

    def test_each_color_is_tuple_of_3(self) -> None:
        colors = get_drive_colors()
        for c in colors:
            assert isinstance(c, tuple)
            assert len(c) == 3

    def test_values_in_range(self) -> None:
        colors = get_drive_colors()
        for r, g, b in colors:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_colorblind_mode_changes_palette(self) -> None:
        set_colorblind_mode(False)
        normal = get_drive_colors()
        set_colorblind_mode(True)
        cb = get_drive_colors()
        set_colorblind_mode(False)
        assert normal != cb


class TestGetFrictionColors:
    def test_torque_history_constants_returns_list(self) -> None:
        colors = get_friction_colors()
        assert isinstance(colors, list)

    def test_torque_history_constants_non_empty(self) -> None:
        assert len(get_friction_colors()) > 0

    def test_colorblind_mode(self) -> None:
        set_colorblind_mode(False)
        normal = get_friction_colors()
        set_colorblind_mode(True)
        cb = get_friction_colors()
        set_colorblind_mode(False)
        assert normal != cb


class TestGetTotalColors:
    def test_torque_history_constants_returns_list(self) -> None:
        colors = get_total_colors()
        assert isinstance(colors, list)

    def test_torque_history_constants_non_empty(self) -> None:
        assert len(get_total_colors()) > 0

    def test_colorblind_mode(self) -> None:
        set_colorblind_mode(False)
        normal = get_total_colors()
        set_colorblind_mode(True)
        cb = get_total_colors()
        set_colorblind_mode(False)
        assert normal != cb


class TestSetColorblindMode:
    def test_off_palette_is_different_from_on_palette(self) -> None:
        set_colorblind_mode(False)
        normal = get_drive_colors()
        set_colorblind_mode(True)
        cb = get_drive_colors()
        set_colorblind_mode(False)  # restore
        assert normal != cb

    def test_false_gives_same_palette_each_time(self) -> None:
        set_colorblind_mode(False)
        colors1 = get_drive_colors()
        set_colorblind_mode(False)
        colors2 = get_drive_colors()
        set_colorblind_mode(False)  # restore
        assert colors1 == colors2

    def test_true_gives_same_palette_each_time(self) -> None:
        set_colorblind_mode(True)
        colors1 = get_drive_colors()
        set_colorblind_mode(True)
        colors2 = get_drive_colors()
        set_colorblind_mode(False)  # restore
        assert colors1 == colors2
