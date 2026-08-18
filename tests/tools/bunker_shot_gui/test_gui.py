"""The Qt shell of the BunkerShot3D designer workbench (issue #8618).

The widget it replaced drew ``np.random.normal`` particles and reported
``0.5 * 0.3 * v**2`` as "Est. Force"; those tests pinned the fake and are
gone. What is pinned now is that the shell renders the real F0 solver's
output and that the validity verdict is impossible to miss.

Qt is imported through ``pytest.importorskip`` because PyQt6 fails to load on
some development machines (an MSVC-runtime mismatch, not a PyQt fault). The
whole computation is still covered there by
``tests/unit/tools/bunker_shot_gui``, which imports no Qt at all.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from bunkershot3d.solvers import EnvelopeStatus  # noqa: E402
from src.tools.bunker_shot_gui.design import (  # noqa: E402
    SolverSetup,
    WedgeDesign,
)
from src.tools.bunker_shot_gui.gui import (  # noqa: E402
    BunkerShotWidget,
    BunkerShotWindow,
    get_dockable_ui,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel  # noqa: E402
from src.tools.bunker_shot_gui.report import status_colour, status_headline  # noqa: E402
from src.tools.bunker_shot_gui.widgets import (  # noqa: E402
    ConditionPanel,
    DesignPanel,
    GridMapWidget,
    VerdictBanner,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: The coarsest settings the geometry package accepts. Real solves, cheap.
COARSE = SolverSetup(
    n_profile_points=12, n_stations=5, playability_points=2, target_carry_m=12.0
)


def _coarse_model(_settings: SolverSetup) -> WorkbenchModel:
    """Factory that ignores the panel's resolution and runs the cheap one."""
    return WorkbenchModel(COARSE)


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """One offscreen QApplication for the module."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


@pytest.fixture()
def widget() -> BunkerShotWidget:
    """A workbench wired to a cheap but genuine model."""
    return BunkerShotWidget(model_factory=_coarse_model)


class TestConstruction:
    def test_the_widget_builds_headless(self, widget: BunkerShotWidget) -> None:
        assert widget.report_text
        assert widget.verdict_text

    def test_nothing_is_claimed_before_a_run(self, widget: BunkerShotWidget) -> None:
        assert "No shot has been run yet" in widget.verdict_text

    def test_the_idle_text_explains_the_envelope(
        self, widget: BunkerShotWidget
    ) -> None:
        text = widget.report_text
        assert "validity verdict" in text
        assert "refuses" in text

    def test_no_procedural_preview_is_advertised(
        self, widget: BunkerShotWidget
    ) -> None:
        for phrase in ("Chrono DEM", "Preview only", "procedural preview"):
            assert phrase not in widget.report_text


class TestRunningADesign:
    def test_running_reports_the_real_solver_numbers(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_design_a()
        text = widget.report_text
        assert "F0 dynamic RFT" in text
        assert "Peak resultant force" in text
        assert "Carry" in text

    def test_the_verdict_banner_leads_with_the_status(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_design_a()
        assert widget.verdict_text == status_headline(EnvelopeStatus.BEYOND_VALIDATION)

    def test_the_maps_are_populated(self, widget: BunkerShotWidget) -> None:
        widget.run_design_a()
        assert widget._bounce_map_a.values.size > 0
        assert widget._window_map_a.values.size > 0

    def test_the_second_column_stays_empty_for_a_single_run(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_design_a()
        assert widget._bounce_map_b.values.size == 0
        assert widget._window_map_b.values.size == 0

    def test_the_run_button_is_wired(self, widget: BunkerShotWidget) -> None:
        widget._run_button.click()
        assert "Peak resultant force" in widget.report_text


class TestRefusalIsUnmissable:
    def _refuse(self, widget: BunkerShotWidget) -> None:
        """Switch the DRFT inertial term off, which the envelope refuses."""
        widget._conditions._dynamic.setChecked(False)
        widget.run_design_a()

    def test_the_banner_turns_red_and_says_refused(
        self, widget: BunkerShotWidget
    ) -> None:
        self._refuse(widget)
        assert widget.verdict_text == status_headline(EnvelopeStatus.REFUSED)
        assert status_colour(EnvelopeStatus.REFUSED) in widget._banner.styleSheet()

    def test_no_force_or_carry_is_shown(self, widget: BunkerShotWidget) -> None:
        self._refuse(widget)
        text = widget.report_text
        assert "No numbers are reported" in text
        assert "Peak resultant force" not in text

    def test_the_maps_are_cleared_rather_than_left_stale(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_design_a()
        assert widget._bounce_map_a.values.size > 0
        self._refuse(widget)
        assert widget._bounce_map_a.values.size == 0
        assert widget._window_map_a.values.size == 0


class TestInputErrors:
    def test_an_impossible_sole_is_reported_as_an_input_error(
        self, widget: BunkerShotWidget
    ) -> None:
        widget._design_a._heel_relief.setValue(0.6)
        widget.run_design_a()
        assert "INPUT ERROR" in widget.verdict_text
        assert "not a solver verdict" in widget.report_text

    def test_an_input_error_clears_the_maps(self, widget: BunkerShotWidget) -> None:
        widget.run_design_a()
        widget._design_a._heel_relief.setValue(0.6)
        widget.run_design_a()
        assert widget._bounce_map_a.values.size == 0

    def test_two_designs_with_one_name_are_refused_before_solving(
        self, widget: BunkerShotWidget
    ) -> None:
        widget._design_b._name.setText(widget._design_a._name.text())
        widget.run_comparison()
        assert "INPUT ERROR" in widget.verdict_text
        assert "different names" in widget.report_text


class TestComparison:
    def test_comparing_fills_both_columns(self, widget: BunkerShotWidget) -> None:
        widget.run_comparison()
        assert widget._bounce_map_a.values.size > 0
        assert widget._bounce_map_b.values.size > 0

    def test_the_comparison_report_ranks_the_two(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_comparison()
        text = widget.report_text
        assert "A/B:" in text
        assert "Leader" in text or "not available" in text

    def test_the_compare_button_is_wired(self, widget: BunkerShotWidget) -> None:
        widget._compare_button.click()
        assert "A/B:" in widget.report_text

    def test_the_banner_shows_the_worse_of_the_two_verdicts(
        self, widget: BunkerShotWidget
    ) -> None:
        widget.run_comparison()
        assert widget.verdict_text == status_headline(EnvelopeStatus.BEYOND_VALIDATION)


class TestDesignPanel:
    def test_the_panel_reads_out_the_w2_design_vector(self) -> None:
        panel = DesignPanel("A", "left", "sm9_58_m")
        design = panel.design()
        assert isinstance(design, WedgeDesign)
        assert design.name == "left"
        assert design.grind_preset == "sm9_58_m"
        assert design.marketed_bounce_deg == pytest.approx(8.0)

    def test_changing_the_preset_reloads_every_control(self) -> None:
        panel = DesignPanel("A", "left", "sm9_58_m")
        before = panel.design()
        panel.load_preset("sm9_54_f")
        after = panel.design()
        assert after.loft_deg == pytest.approx(54.0)
        assert after.loft_deg != pytest.approx(before.loft_deg)

    def test_an_edited_control_reaches_the_design(self) -> None:
        panel = DesignPanel("A", "left", "sm9_58_m")
        panel._sole_width.setValue(23.0)
        assert panel.design().sole_width_mm == pytest.approx(23.0)


class TestConditionPanel:
    def test_the_sand_condition_is_read_out(self) -> None:
        panel = ConditionPanel()
        panel._firmness.setValue(2.0)
        condition = panel.sand_condition()
        assert condition.firmness_kg_per_cm2 == pytest.approx(2.0)

    def test_the_swing_is_read_out_in_si(self) -> None:
        panel = ConditionPanel()
        panel._entry.setValue(120.0)
        swing = panel.swing_setup()
        assert swing.entry_distance_behind_ball_m == pytest.approx(0.120)

    def test_the_attack_angle_control_cannot_reach_a_level_blow(self) -> None:
        panel = ConditionPanel()
        panel._attack.setValue(5.0)
        assert panel.swing_setup().attack_angle_deg < 0.0

    def test_the_dynamic_term_switch_is_read_out(self) -> None:
        panel = ConditionPanel()
        panel._dynamic.setChecked(False)
        assert panel.swing_setup().dynamic_terms_active is False

    def test_the_study_settings_are_read_out(self) -> None:
        panel = ConditionPanel()
        panel._grid.setValue(4)
        panel._target_carry.setValue(9.0)
        settings = panel.solver_setup()
        assert settings.playability_points == 4
        assert settings.target_carry_m == pytest.approx(9.0)


class TestVerdictBanner:
    def test_the_idle_state_claims_nothing(self) -> None:
        assert "No shot has been run yet" in VerdictBanner().text()

    @pytest.mark.parametrize("status", list(EnvelopeStatus))
    def test_every_status_paints_its_own_colour(self, status: EnvelopeStatus) -> None:
        banner = VerdictBanner()
        banner.show_status(status)
        assert status_colour(status) in banner.styleSheet()
        assert banner.text() == status_headline(status)

    def test_an_input_error_is_not_dressed_up_as_a_verdict(self) -> None:
        banner = VerdictBanner()
        banner.show_error("bad sole")
        assert banner.text().startswith("INPUT ERROR")


class TestGridMapWidget:
    def test_a_new_map_holds_nothing(self) -> None:
        assert GridMapWidget("m").values.size == 0

    def test_a_grid_round_trips(self) -> None:
        widget = GridMapWidget("m")
        widget.set_grid(np.arange(6.0).reshape(2, 3), caption="hello")
        assert widget.values.shape == (2, 3)
        assert widget.caption == "hello"

    def test_clearing_drops_the_grid_and_the_caption(self) -> None:
        widget = GridMapWidget("m")
        widget.set_grid(np.zeros((2, 2)), caption="hello")
        widget.clear()
        assert widget.values.size == 0
        assert widget.caption == ""

    def test_a_one_dimensional_grid_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2-D array"):
            GridMapWidget("m").set_grid(np.zeros(4))

    def test_a_mismatched_mask_is_refused(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            GridMapWidget("m").set_grid(np.zeros((2, 2)), mask=np.zeros((3, 3), bool))

    def test_the_title_is_kept(self) -> None:
        assert GridMapWidget("bounce").title == "bounce"


class TestWindow:
    def test_the_window_titles_itself_as_a_design_tool(self) -> None:
        window = BunkerShotWindow()
        assert isinstance(window, QMainWindow)
        assert "Workbench" in window.windowTitle()
        assert isinstance(window.centralWidget(), BunkerShotWidget)

    def test_the_status_bar_states_the_refusal_policy(self) -> None:
        window = BunkerShotWindow()
        assert "refused" in window.statusBar().currentMessage()

    def test_closing_cleans_the_widget_up(self) -> None:
        window = BunkerShotWindow()
        window.close()

    def test_get_dockable_ui_returns_a_window(self) -> None:
        ui = get_dockable_ui()
        assert isinstance(ui, BunkerShotWindow)

    def test_cleanup_is_idempotent(self, widget: BunkerShotWidget) -> None:
        widget.cleanup()
        widget.cleanup()


class TestEmbedAdapter:
    """The launcher-facing half of the contract, where Qt is allowed."""

    def test_the_adapter_builds_the_workbench_widget(self) -> None:
        from src.tools.bunker_shot_gui import BunkerShotGuiAdapter

        adapter = BunkerShotGuiAdapter()
        built = adapter.create_main_widget(None)
        assert isinstance(built, BunkerShotWidget)
        adapter.cleanup()
        assert adapter.is_dirty() is False
