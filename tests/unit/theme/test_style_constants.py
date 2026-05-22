"""Tests for src.shared.python.theme.style_constants (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.theme.style_constants import Styles


def _declarations(style: str) -> set[str]:
    return {declaration.strip() for declaration in style.split(";") if declaration.strip()}


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

    def test_bold_status_constants_extend_base_status_colors(self) -> None:
        assert Styles.STATUS_SUCCESS_BOLD.startswith(Styles.STATUS_SUCCESS)
        assert Styles.STATUS_ERROR_BOLD.startswith(Styles.STATUS_ERROR)
        assert Styles.STATUS_INACTIVE_BOLD.startswith(Styles.STATUS_INACTIVE)

    def test_simple_status_colors_are_resettable(self) -> None:
        assert Styles.COLOR_GREEN == "color: green;"
        assert Styles.COLOR_RED == "color: red;"
        assert Styles.COLOR_ORANGE == "color: orange;"
        assert Styles.COLOR_GRAY == "color: gray;"
        assert Styles.COLOR_RESET == ""


class TestButtonConstants:
    def test_btn_run_is_string(self) -> None:
        assert isinstance(Styles.BTN_RUN, str)

    def test_btn_stop_is_string(self) -> None:
        assert isinstance(Styles.BTN_STOP, str)

    def test_btn_primary_is_string(self) -> None:
        assert isinstance(Styles.BTN_PRIMARY, str)

    def test_btn_danger_is_string(self) -> None:
        assert isinstance(Styles.BTN_DANGER, str)

    def test_button_constants_include_expected_states(self) -> None:
        assert "QPushButton:hover" in Styles.BTN_PRIMARY
        assert "QPushButton:hover" in Styles.BTN_SECONDARY
        assert "QPushButton:checked" in Styles.BTN_AI_CHAT
        assert "QPushButton:disabled" in Styles.BTN_LAUNCH_READY
        assert "QPushButton:hover:!disabled" in Styles.BTN_LAUNCH_READY
        assert "QPushButton:checked" in Styles.BTN_LAYOUT_TOGGLE
        assert "QPushButton:checked" in Styles.BTN_RECORD_CHECKED
        assert "QPushButton:disabled" in Styles.BTN_SEND

    def test_button_semantic_colors_remain_distinct(self) -> None:
        assert "#4CAF50" in Styles.BTN_RUN
        assert "#f44336" in Styles.BTN_STOP
        assert "#0A84FF" in Styles.BTN_PRIMARY
        assert "#d62728" in Styles.BTN_DANGER


class TestLayoutConstants:
    def test_spacing_constants_are_positive_integers(self) -> None:
        assert Styles.SIDEBAR_MIN_WIDTH == 120
        assert Styles.SPACING_LG > Styles.SPACING_MD > Styles.SPACING_SM > 0
        assert Styles.MARGIN_PAGE > Styles.SPACING_LG

    def test_statusbar_aliases_reuse_canonical_text_constants(self) -> None:
        assert Styles.STATUSBAR_TIME is Styles.TEXT_SUCCESS_PADDED
        assert Styles.STATUSBAR_STATE_RUNNING is Styles.TEXT_SUCCESS_PADDED

    def test_transparent_container_constants_use_qt_selectors(self) -> None:
        assert Styles.TRANSPARENT_BG == "background: transparent;"
        assert Styles.SCROLL_AREA_TRANSPARENT.startswith("QScrollArea")
        assert Styles.SPLITTER_HANDLE.startswith("QSplitter::handle")
        assert Styles.LABEL_TRANSPARENT.startswith("QLabel")


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

    def test_color_swatch_formats_compact_rgb_and_border(self) -> None:
        result = Styles.color_swatch(1, 2, 3)

        assert result == "background-color: rgb(1,2,3); border: 1px solid #555;"

    def test_color_swatch_passes_edge_values_through(self) -> None:
        result = Styles.color_swatch(0, 255, -1)

        assert "rgb(0,255,-1)" in result

    def test_status_chip_formats_background_text_and_shape(self) -> None:
        result = Styles.status_chip("transparent", "rgb(1, 2, 3)")

        assert _declarations(result) == {
            "background-color: transparent",
            "color: rgb(1, 2, 3)",
            "padding: 2px 6px",
            "border-radius: 4px",
        }

    def test_colored_bold_formats_exact_declarations(self) -> None:
        result = Styles.colored_bold("var(--accent)")

        assert _declarations(result) == {
            "color: var(--accent)",
            "font-weight: bold",
        }

    def test_no_image_label_formats_transparent_italic_qss(self) -> None:
        result = Styles.no_image_label("rgba(255, 255, 255, 0.5)")

        assert result.startswith("QLabel {")
        assert result.endswith("}")
        assert "color: rgba(255, 255, 255, 0.5);" in result
        assert "font-style: italic;" in result
        assert "border: none;" in result
        assert "background: transparent;" in result
