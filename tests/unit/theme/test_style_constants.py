"""Tests for src.shared.python.theme.style_constants (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.theme.style_constants import Styles


class TestStatusConstants:
    def test_status_success_is_string(self) -> None:
        assert isinstance(Styles.STATUS_SUCCESS, str)

    def test_status_error_is_string(self) -> None:
        assert isinstance(Styles.STATUS_ERROR, str)

    def test_status_warning_is_string(self) -> None:
        assert isinstance(Styles.STATUS_WARNING, str)

    def test_status_info_is_string(self) -> None:
        assert isinstance(Styles.STATUS_INFO, str)

    def test_status_success_contains_color(self) -> None:
        assert "color" in Styles.STATUS_SUCCESS

    def test_status_success_bold_contains_bold(self) -> None:
        assert "bold" in Styles.STATUS_SUCCESS_BOLD

    def test_status_error_bold_contains_bold(self) -> None:
        assert "bold" in Styles.STATUS_ERROR_BOLD


class TestButtonConstants:
    def test_btn_run_is_string(self) -> None:
        assert isinstance(Styles.BTN_RUN, str)

    def test_btn_stop_is_string(self) -> None:
        assert isinstance(Styles.BTN_STOP, str)

    def test_btn_primary_is_string(self) -> None:
        assert isinstance(Styles.BTN_PRIMARY, str)

    def test_btn_danger_is_string(self) -> None:
        assert isinstance(Styles.BTN_DANGER, str)


class TestHelperMethods:
    def test_color_swatch_returns_string(self) -> None:
        result = Styles.color_swatch(255, 0, 0)
        assert isinstance(result, str)

    def test_color_swatch_contains_rgb(self) -> None:
        result = Styles.color_swatch(100, 200, 50)
        assert "100" in result
        assert "200" in result
        assert "50" in result

    def test_status_chip_returns_string(self) -> None:
        result = Styles.status_chip("#FF0000", "#FFFFFF")
        assert isinstance(result, str)

    def test_status_chip_contains_colors(self) -> None:
        result = Styles.status_chip("#FF0000", "#FFFFFF")
        assert "#FF0000" in result
        assert "#FFFFFF" in result

    def test_colored_bold_returns_string(self) -> None:
        result = Styles.colored_bold("#123456")
        assert isinstance(result, str)

    def test_colored_bold_contains_color(self) -> None:
        result = Styles.colored_bold("#AABBCC")
        assert "#AABBCC" in result

    def test_no_image_label_returns_string(self) -> None:
        result = Styles.no_image_label("#999999")
        assert isinstance(result, str)
