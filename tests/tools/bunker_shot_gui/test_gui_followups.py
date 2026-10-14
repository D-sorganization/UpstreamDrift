"""Tests for BunkerShot3D GUI follow-ups (issue #12536).

Pins four workbench requirements:
- PyQt control inventory: every control either takes effect or shows a disabled reason
  both before and after evaluation/load.
- Desktop save/reload of a workbench evaluation round-trips correctly and populates GUI panels.
- Loading corrupted records fails closed with ExportIntegrityError.
- Export views generates stamped figure files that validate against read_export.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.bunker_shot_gui.design import SolverSetup  # noqa: E402
from src.tools.bunker_shot_gui.export import (  # noqa: E402
    ExportIntegrityError,
    read_export,
)
from src.tools.bunker_shot_gui.gui import (  # noqa: E402
    BunkerShotWidget,
    BunkerShotWindow,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

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


def _pump_until(predicate: Any, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    QCoreApplication.processEvents()
    return predicate()


def test_control_inventory_and_disabled_reasons_before_and_after_eval(
    widget: BunkerShotWidget,
) -> None:
    window = BunkerShotWindow(widget=widget)

    # Initial state before any evaluation
    assert widget.control_enabled("run_design_a") is True
    assert widget.disabled_reasons()["run_design_a"] is None

    assert widget.control_enabled("compare") is True
    assert widget.disabled_reasons()["compare"] is None

    assert widget.control_enabled("cross_tier") is True
    assert widget.disabled_reasons()["cross_tier"] is None

    assert widget.control_enabled("load_evaluation") is True
    assert widget.disabled_reasons()["load_evaluation"] is None

    assert widget.control_enabled("export_views") is False
    assert "no evaluation" in widget.disabled_reasons()["export_views"]

    assert widget.control_enabled("save_evaluation") is False
    assert "no evaluation" in widget.disabled_reasons()["save_evaluation"]

    assert window._export_action.isEnabled() is False
    assert window._save_action.isEnabled() is False
    assert window._load_action.isEnabled() is True

    # Run nominal solve
    widget.run_design_a()
    assert _pump_until(lambda: widget.current_evaluation is not None)

    # State after evaluation
    assert widget.control_enabled("export_views") is True
    assert widget.disabled_reasons()["export_views"] is None

    assert widget.control_enabled("save_evaluation") is True
    assert widget.disabled_reasons()["save_evaluation"] is None

    assert window._export_action.isEnabled() is True
    assert window._save_action.isEnabled() is True


def test_gui_save_and_load_round_trip(widget: BunkerShotWidget, tmp_path: Path) -> None:
    widget.run_design_a()
    assert _pump_until(lambda: widget.current_evaluation is not None)
    eval_before = widget.current_evaluation
    assert eval_before is not None

    save_path = tmp_path / "saved_eval.json"
    saved = widget.save_evaluation(save_path)
    assert saved == save_path
    assert save_path.is_file()

    # Modify the condition panel to a different condition
    widget._conditions._condition.setCurrentText("soft")

    # Load back the evaluation
    record = widget.load_evaluation(save_path)
    assert record["schema_version"] == "bunker-workbench-evaluation/1"
    assert widget._loaded_record is not None
    assert widget._loaded_record["digest"] == record["digest"]

    # Verify condition panel was updated from the record
    assert widget._conditions._condition.currentText() == "firm"


def test_gui_load_rejects_corrupted_file(
    widget: BunkerShotWidget, tmp_path: Path
) -> None:
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid json", encoding="utf-8")

    with pytest.raises(ExportIntegrityError):
        widget.load_evaluation(corrupt_file)


def test_gui_load_rejects_tampered_digest(
    widget: BunkerShotWidget, tmp_path: Path
) -> None:
    widget.run_design_a()
    assert _pump_until(lambda: widget.current_evaluation is not None)

    save_path = tmp_path / "eval_to_tamper.json"
    widget.save_evaluation(save_path)

    data = json.loads(save_path.read_text(encoding="utf-8"))
    data["shot"]["peak_force_n"] = 99999.0
    save_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ExportIntegrityError, match="digest"):
        widget.load_evaluation(save_path)


def test_gui_export_views(widget: BunkerShotWidget, tmp_path: Path) -> None:
    widget.run_design_a()
    assert _pump_until(lambda: widget.current_evaluation is not None)

    export_dir = tmp_path / "views"
    exported_figures = widget.export_views(export_dir)

    assert len(exported_figures) >= 1
    for figure in exported_figures:
        assert figure.path.is_file()
        reloaded = read_export(figure.path)
        assert reloaded.stamp is not None
        assert reloaded.stamp.status is not None
        assert reloaded.stamp.tier is not None
