"""#8880 proof-of-migration: the workbench's solver calls no longer block
the GUI thread behind a wait cursor alone.

``bunker_shot_gui`` was named explicitly in issue #8880 as remaining work:
its ``_guarded`` call sites set a wait cursor but still ran the F0 solver
(and, for cross-tier, the F1 continuum march -- minutes, not milliseconds)
inline in a ``clicked`` handler. These tests assert the migration's
deliverables: the solver call runs off the GUI thread, all three trigger
buttons are disabled for the duration of any one run, and the synchronous
``run_design_a``/``run_comparison``/``run_cross_tier`` core is unchanged
(it is what the rest of this package's test suite drives).
"""

from __future__ import annotations

import sys
import time

import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

from PyQt6.QtCore import QCoreApplication, QThread  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.design import SolverSetup  # noqa: E402
from src.tools.bunker_shot_gui.gui import BunkerShotWidget  # noqa: E402
from src.tools.bunker_shot_gui.model import WorkbenchModel  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: The coarsest settings the geometry package accepts. Real solves, cheap.
COARSE = SolverSetup(
    n_profile_points=12, n_stations=5, playability_points=2, target_carry_m=12.0
)


def _coarse_model(_settings: SolverSetup) -> WorkbenchModel:
    return WorkbenchModel(COARSE)


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv[:1])
    return application


@pytest.fixture()
def widget():  # noqa: ANN201
    the_widget = BunkerShotWidget(model_factory=_coarse_model)
    yield the_widget
    the_widget.cleanup()


def _pump_until(predicate, timeout_s: float = 20.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    QCoreApplication.processEvents()
    return predicate()


def test_run_design_a_async_runs_off_the_gui_thread(widget: BunkerShotWidget) -> None:
    gui_thread = QThread.currentThread()
    seen: list[object] = []
    original_evaluate = WorkbenchModel.evaluate

    def _record(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        seen.append(QThread.currentThread())
        return original_evaluate(self, *args, **kwargs)

    WorkbenchModel.evaluate = _record  # type: ignore[method-assign]
    try:
        widget.run_design_a_async()
        assert _pump_until(lambda: "Peak resultant force" in widget.report_text)
    finally:
        WorkbenchModel.evaluate = original_evaluate  # type: ignore[method-assign]

    assert seen, "design A never ran"
    assert seen[0] is not gui_thread, "design A still runs on the GUI thread"


def test_all_trigger_buttons_are_disabled_while_an_action_runs(
    widget: BunkerShotWidget,
) -> None:
    release: list[bool] = []
    original_evaluate = WorkbenchModel.evaluate

    def _blocking(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        while not release:
            QThread.msleep(5)
        return original_evaluate(self, *args, **kwargs)

    WorkbenchModel.evaluate = _blocking  # type: ignore[method-assign]
    try:
        widget.run_design_a_async()
        assert _pump_until(lambda: not widget._run_button.isEnabled())
        assert not widget._compare_button.isEnabled()
        assert not widget._cross_tier_button.isEnabled()
        assert widget.action_bar.cancel_button.isEnabled()

        release.append(True)
        assert _pump_until(lambda: widget._run_button.isEnabled())
    finally:
        WorkbenchModel.evaluate = original_evaluate  # type: ignore[method-assign]


def test_synchronous_run_design_a_still_works(widget: BunkerShotWidget) -> None:
    """The sync core (used by the rest of this package's tests) is unchanged."""
    widget.run_design_a()
    assert "Peak resultant force" in widget.report_text
