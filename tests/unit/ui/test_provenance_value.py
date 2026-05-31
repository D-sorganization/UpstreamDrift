"""Unit tests for the ProvenanceValueLabel PyQt6 wrapper (epic #5968).

The label consumes the *existing*
:class:`~src.shared.python.ux.provenance.ProvenanceValue` dataclass
(DRY) and renders the value text with a "linked" affordance: the
provenance description is exposed via tooltip + whatsThis so a hover or
right-click answers "why does this say 500?".

Single-widget Qt usage to respect the multi-widget segfault constraint.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from src.shared.python.ux.provenance import (  # noqa: E402
    ProvenanceRecord,
    ProvenanceValue,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def qt_app():
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as e:  # noqa: F841
        pytest.skip(f"PyQt6 runtime unavailable: {e}")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


def _sample_value() -> ProvenanceValue:
    record = ProvenanceRecord(
        formula="fps = 1.0 / timestep",
        inputs=("simulation.timestep",),
        source="mujoco:run-42",
        computed_at=datetime(2026, 5, 30, 12, 0, tzinfo=timezone.utc),
        engine="mujoco",
        run_id="run-42",
    )
    return ProvenanceValue(
        value=500.0, record=record, display_units="fps", label="Frame rate"
    )


def test_label_renders_value_and_units(qt_app) -> None:
    from src.shared.python.ui.provenance_value import ProvenanceValueLabel

    widget = ProvenanceValueLabel(_sample_value())
    assert "500" in widget.text()
    assert "fps" in widget.text()


def test_label_tooltip_carries_provenance(qt_app) -> None:
    from src.shared.python.ui.provenance_value import ProvenanceValueLabel

    pv = _sample_value()
    widget = ProvenanceValueLabel(pv)
    tip = widget.toolTip()
    assert "fps = 1.0 / timestep" in tip
    assert "simulation.timestep" in tip
    assert "mujoco:run-42" in tip
    # describe() is the single source for the popover text (DRY).
    assert widget.toolTip() == pv.describe()


def test_label_marks_linked_when_inputs_present(qt_app) -> None:
    from src.shared.python.ui.provenance_value import ProvenanceValueLabel

    widget = ProvenanceValueLabel(_sample_value())
    assert widget.is_linked() is True


def test_label_not_linked_for_constant(qt_app) -> None:
    from src.shared.python.ui.provenance_value import ProvenanceValueLabel

    record = ProvenanceRecord(
        formula="constant 9.81",
        inputs=(),
        source="internal:g",
        computed_at=datetime(2026, 5, 30, tzinfo=timezone.utc),
        engine="mujoco",
        run_id="static",
    )
    pv = ProvenanceValue(value=9.81, record=record, display_units="m/s^2")
    widget = ProvenanceValueLabel(pv)
    assert widget.is_linked() is False


def test_rejects_non_provenance_value(qt_app) -> None:
    from src.shared.python.ui.provenance_value import ProvenanceValueLabel

    with pytest.raises(TypeError):
        ProvenanceValueLabel(object())  # type: ignore[arg-type]
