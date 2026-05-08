"""Tests for the Overview-tab metadata tree + provenance summary."""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_real_capture_file_when_available(qt_app) -> None:
    """Smoke against the canonical sample C3D when present."""
    here = Path(__file__).resolve()
    repo_root = here.parents[5]
    sample = repo_root / "data" / "C3D_TA_Driver.c3d"
    if not sample.is_file():
        pytest.skip("sample C3D not present in this checkout")
    from src.apps.services.c3d_loader import load_c3d_file  # type: ignore
    from src.apps.ui.tabs.overview_tab import OverviewTab  # type: ignore

    model = load_c3d_file(str(sample))
    tab = OverviewTab()
    tab.update_from_model(model)
    # ezc3d guarantees POINT and ANALOG groups; real captures include several
    # vendor-specific groups too.
    assert tab.tree_group_count >= 5
    assert tab.tree_node_count > 50
