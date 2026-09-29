"""Tests for the Overview-tab metadata tree + provenance summary."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from ._viewer_test_helpers import make_synthetic_model

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


def _wrap(value):
    """Wrap a Python value in the ezc3d-style ``{"value": ...}`` envelope."""
    return {"value": np.asarray(value)}


def test_metadata_tree_with_synthetic_params(qt_app) -> None:
    from src.apps.ui.tabs.overview_tab import OverviewTab  # type: ignore

    model = make_synthetic_model(["WaistLeft", "WaistRight"], n_frames=10)
    model.raw_parameters = {
        "POINT": {
            "USED": _wrap([2]),
            "FRAMES": _wrap([10]),
            "RATE": _wrap([100.0]),
            "UNITS": _wrap(["m"]),
            "X_SCREEN": _wrap(["+X"]),
            "Y_SCREEN": _wrap(["+Z"]),
            "LABELS": _wrap(["WaistLeft", "WaistRight"]),
        },
        "ANALOG": {"USED": _wrap([0]), "RATE": _wrap([1000.0])},
        "FORCE_PLATFORM": {"USED": _wrap([2])},
        "TRIAL": {"ACTUAL_START_FIELD": _wrap([1])},
        "MANUFACTURER": {
            "SOFTWARE": _wrap(["GenericMocap"]),
            "VERSION": _wrap(["1.0"]),
        },
        "VENDOR_X": {
            "CAPTURE_ID": _wrap(["abc-123"]),
            "SUB_TYPE": _wrap(["full_swing"]),
            "PLAYER_ID": _wrap(["P-001"]),
        },
    }
    tab = OverviewTab()
    tab.update_from_model(model)
    # 5 elevated groups + VENDOR_X.
    assert tab.tree_group_count >= 5


def test_no_provenance_placeholder(qt_app) -> None:
    from src.apps.ui.tabs.overview_tab import OverviewTab  # type: ignore

    model = make_synthetic_model(["WaistLeft", "WaistRight"], n_frames=5)
    model.raw_parameters = {
        "POINT": {
            "FRAMES": _wrap([5]),
            "RATE": _wrap([100.0]),
            "UNITS": _wrap(["m"]),
        },
    }
    tab = OverviewTab()
    tab.update_from_model(model)
    assert tab.tree_group_count >= 1


def test_metadata_panel_provenance_toggle(qt_app) -> None:
    from src.apps.ui.tabs.overview_tab import OverviewTab  # type: ignore

    model = make_synthetic_model(["WaistLeft", "WaistRight"], n_frames=10)
    model.raw_parameters = {
        "POINT": {
            "USED": _wrap([2]),
            "FRAMES": _wrap([10]),
            "RATE": _wrap([100.0]),
            "UNITS": _wrap(["m"]),
            "X_SCREEN": _wrap(["+X"]),
            "Y_SCREEN": _wrap(["+Z"]),
        },
        "VENDOR_X": {
            "CAPTURE_ID": _wrap(["abc-123"]),
            "PLAYER_ID": _wrap(["P-001"]),
            "SOME_OTHER_KEY": _wrap(["normal_value"]),
        },
    }
    tab = OverviewTab()

    # 1. By default, show provenance is True
    tab.update_from_model(model)
    assert tab.chk_show_provenance.isChecked() is True
    assert tab._prov_group.isHidden() is False

    # 2. Toggle to False
    tab.chk_show_provenance.setChecked(False)
    assert tab._prov_group.isHidden() is True

    # Check key list in the tree to make sure CAPTURE_ID and PLAYER_ID are filtered out
    root = tab._tree_model.invisibleRootItem()
    group_found = False
    for r in range(root.rowCount()):
        group_item = root.child(r, 0)
        if group_item.text() == "VENDOR_X":
            group_found = True
            child_keys = [
                group_item.child(c, 0).text() for c in range(group_item.rowCount())
            ]
            assert "CAPTURE_ID" not in child_keys
            assert "PLAYER_ID" not in child_keys
            assert "SOME_OTHER_KEY" in child_keys
    assert group_found is True
