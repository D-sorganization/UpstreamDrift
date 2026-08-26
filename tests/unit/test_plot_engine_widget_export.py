"""Tombstone for the PlotWidget export-wiring tests (Issue #8828).

The tests originally here covered PlotWidget._export_plot routing through
plotting.export.export_figure and PlotIdentity embedding for the generic
dashboard export path. That wiring lives in
src/shared/python/plot_engine/pyqt6_widget.py, which mirrors
D-sorganization/Tools's src/shared/python/plot_engine/ 1:1 (a Tools-owned
child copy per tests/unit/repo_hygiene/test_tools_child_copy_contract.py)
-- this repo must not edit it directly, so the widget-level wiring was
reverted out of the #8828 PR. The identity/export-metadata plumbing itself
(plotting/identity.py, plotting/export.py) is covered by
tests/unit/plotting/test_identity.py and tests/unit/plotting/test_export.py.

Once the widget wiring lands in Tools and flows here via a vendor/ud-tools
pin bump, restore end-to-end coverage here (or in the vendored tool's own
test suite) for PlotWidget.set_identity()/_export_plot.
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_placeholder_widget_export_wiring_pending_tools_pin() -> None:
    """Tombstone: real coverage returns once the Tools-owned wiring lands."""
