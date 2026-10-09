"""Offscreen PyQt6 tests for the ground-reaction plot widget (GCV-5, #11711)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from src.shared.python.biomechanics.ground_reaction import (
    ContactSet,
    GroundReactionSeries,
    analyze_ground_reaction,
)
from src.shared.python.biomechanics.ground_reaction_plot_model import (
    build_ground_reaction_plot_series,
    plot_series_to_json,
    unavailable_ground_reaction_plot,
)

pytestmark = pytest.mark.unit

_APP = None


def _qt():
    global _APP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyQt6 not loadable: {exc}")
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _settle() -> None:
    """Run queued ``draw_idle`` callbacks while the widget is still alive.

    Otherwise the widget is garbage-collected with a pending single-shot draw
    that fires on the deleted canvas during a later test's event processing.
    """
    _APP.processEvents()


def _plot():
    left = ContactSet(np.array([[0.0, 0.0, 700.0]]), np.array([[0.0, 0.15, 0.0]]))
    b = analyze_ground_reaction({"left": left}, (0.0, 0.0, 0.95))
    series = GroundReactionSeries.from_breakdowns([0.0, 0.1], [b, b])
    return build_ground_reaction_plot_series(series, events={"impact": 0.1})


def test_widget_starts_unavailable_and_draws_six_panels() -> None:
    _qt()
    from src.tools.ground_reaction_plots.gui import GroundReactionPlotWidget

    w = GroundReactionPlotWidget()
    assert "unavailable" in w.status_label.text()
    w.set_series(_plot())
    assert w.status_label.text() == "Feet: left | 2 samples"
    assert len(w.figure.axes) == 6
    w.set_series(unavailable_ground_reaction_plot("no GRF in Simscape"))
    assert w.status_label.text() == "unavailable: no GRF in Simscape"
    _settle()
    w.cleanup()
    w.cleanup()  # idempotent


def test_widget_loads_the_api_payload_json(tmp_path) -> None:
    _qt()
    from src.tools.ground_reaction_plots.gui import GroundReactionPlotWidget

    path = tmp_path / "grf.json"
    path.write_text(plot_series_to_json(_plot()), encoding="utf-8")
    w = GroundReactionPlotWidget()
    w.load_json(path)
    assert len(w.figure.axes) == 6
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        w.load_json(bad)
    _settle()


def test_embed_adapter_creates_one_widget_and_cleans_up() -> None:
    _qt()
    from src.tools.ground_reaction_plots._embed_adapter import (
        GroundReactionPlotsEmbedAdapter,
    )

    adapter = GroundReactionPlotsEmbedAdapter()
    assert adapter.tool_id == "ground_reaction_plots"
    assert adapter.embed_capabilities().prefers_dock
    widget = adapter.create_main_widget(None)
    assert adapter.create_main_widget(None) is widget
    adapter.cleanup()
    adapter.cleanup()
    assert not adapter.is_dirty()
