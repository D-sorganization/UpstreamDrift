"""End-to-end PyQt6 launch-and-interact tests for Ground Reaction Plots (#15522).

``tests/unit/biomechanics/test_ground_reaction_plot_widget.py`` drives the
widget through its Python API.  These tests drive it the way a user does: they
launch it through the standalone ``main()`` entry point and press the real
"Open Ground Reaction JSON..." / "Import Plate Trial..." buttons with
``QTest`` mouse clicks.  The modal ``QFileDialog`` is replaced by a stub (a
native dialog cannot be driven headlessly); everything after the file choice is
the real code path, asserted on what the user sees: status line, banner and the
six-panel figure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QCoreApplication, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QFileDialog, QPushButton  # noqa: E402

from src.shared.python.biomechanics.ground_reaction import (  # noqa: E402
    ContactSet,
    GroundReactionSeries,
    analyze_ground_reaction,
)
from src.shared.python.biomechanics.ground_reaction_plot_model import (  # noqa: E402
    build_ground_reaction_plot_series,
    plot_series_to_json,
)
from src.tools.ground_reaction_plots import __main__ as grf_main  # noqa: E402
from src.tools.ground_reaction_plots.gui import (  # noqa: E402
    UNAVAILABLE_TEXT,
    GroundReactionPlotWidget,
)

pytestmark = pytest.mark.unit

_APP: QApplication | None = None


def _ensure_qapp() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # keep a module-level reference alive
    return app


def _payload_json() -> str:
    contact = ContactSet(np.array([[0.0, 0.0, 700.0]]), np.array([[0.0, 0.15, 0.0]]))
    breakdown = analyze_ground_reaction({"left": contact}, (0.0, 0.0, 0.95))
    series = GroundReactionSeries.from_breakdowns([0.0, 0.1], [breakdown, breakdown])
    return plot_series_to_json(
        build_ground_reaction_plot_series(series, events={"impact": 0.1})
    )


def _click(button: QPushButton) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QCoreApplication.processEvents()


def _choose(monkeypatch: pytest.MonkeyPatch, path: Path | str) -> list[str]:
    """Make the file dialog "pick" ``path``; return the captions it was shown."""
    captions: list[str] = []

    def _fake(_parent, caption="", *_args, **_kwargs):  # noqa: ANN001, ANN202
        captions.append(caption)
        return str(path), ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(_fake))
    return captions


@pytest.fixture
def window():  # noqa: ANN201
    _ensure_qapp()
    widget = GroundReactionPlotWidget()
    widget.show()
    QCoreApplication.processEvents()
    yield widget
    widget.cleanup()
    widget.close()
    QCoreApplication.processEvents()
    widget.deleteLater()
    QCoreApplication.processEvents()


def test_main_launches_a_visible_titled_window(monkeypatch) -> None:  # noqa: ANN001
    app = _ensure_qapp()
    seen: list[GroundReactionPlotWidget] = []

    def _fake_exec() -> int:
        seen.extend(
            w
            for w in QApplication.topLevelWidgets()
            if isinstance(w, GroundReactionPlotWidget)
        )
        return 0

    monkeypatch.setattr(app, "exec", _fake_exec)
    assert grf_main.main() == 0
    try:
        assert len(seen) == 1
        win = seen[0]
        assert win.isVisible()
        assert win.windowTitle() == "Ground Reaction Plots"
        assert (win.width(), win.height()) == (960, 820)
        assert UNAVAILABLE_TEXT in win.status_label.text()
        assert win.open_button.isEnabled()
        assert win.plate_button.isEnabled()
        assert not win.banner_label.isVisible()
    finally:
        for win in seen:
            win.cleanup()
            win.close()
            win.deleteLater()
        QCoreApplication.processEvents()


def test_open_button_loads_json_and_draws_six_panels(
    window: GroundReactionPlotWidget, tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    path = tmp_path / "grf.json"
    path.write_text(_payload_json(), encoding="utf-8")
    captions = _choose(monkeypatch, path)

    _click(window.open_button)

    assert captions == ["Open Ground Reaction JSON"]
    assert window.status_label.text() == "Feet: left | 2 samples"
    assert len(window.figure.axes) == 6
    assert not window.banner_label.isVisible()


def test_cancelled_dialog_leaves_the_window_unchanged(
    window: GroundReactionPlotWidget, monkeypatch
) -> None:  # noqa: ANN001
    before = window.status_label.text()
    captions = _choose(monkeypatch, "")

    _click(window.open_button)

    assert captions == ["Open Ground Reaction JSON"]
    assert window.status_label.text() == before
    assert len(window.figure.axes) == 0


def test_malformed_json_reports_unavailable_and_keeps_prior_plot(
    window: GroundReactionPlotWidget, tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    good = tmp_path / "good.json"
    good.write_text(_payload_json(), encoding="utf-8")
    _choose(monkeypatch, good)
    _click(window.open_button)
    assert len(window.figure.axes) == 6

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2]), encoding="utf-8")
    _choose(monkeypatch, bad)
    _click(window.open_button)

    assert window.status_label.text().startswith(f"{UNAVAILABLE_TEXT}:")
    assert "object" in window.status_label.text()
    assert len(window.figure.axes) == 6  # the earlier plot is not wiped


def test_missing_file_reports_unavailable(
    window: GroundReactionPlotWidget, tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    _choose(monkeypatch, tmp_path / "nope.json")

    _click(window.open_button)

    assert window.status_label.text().startswith(f"{UNAVAILABLE_TEXT}:")


def test_unqualified_measurement_shows_the_warning_banner(
    window: GroundReactionPlotWidget, tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    payload = json.loads(_payload_json())
    payload["measured_qualified"] = False
    path = tmp_path / "unqualified.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _choose(monkeypatch, path)

    _click(window.open_button)

    assert window.banner_label.isVisible()
    assert "Unqualified" in window.banner_label.text()


def test_plate_import_without_a_model_series_is_refused(
    window: GroundReactionPlotWidget, tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    # A host-declared context skips the declarations dialog, so the click goes
    # straight to the file chooser and then to the real import guard.
    from src.tools.ground_reaction_plots.plate_import_form import PlateImportContext

    window.set_plate_context(
        PlateImportContext(declared=None, placement=None, alignment=None, options=None)  # type: ignore[arg-type]
    )
    captions = _choose(monkeypatch, tmp_path / "trial.json")

    _click(window.plate_button)

    assert captions == ["Import Plate Trial"]
    assert window.status_label.text() == (
        f"{UNAVAILABLE_TEXT}: load a model ground reaction before importing a trial"
    )
