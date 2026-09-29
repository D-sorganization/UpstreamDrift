"""Tests for C3D viewer coordinate axis orientation conventions."""

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
    return {"value": np.asarray(value)}


def test_transform_positions_conventions(qt_app) -> None:
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    model = make_synthetic_model(["Marker1"], n_frames=5)
    model.markers["Marker1"].position = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.2, 2.2, 3.2],
            [1.3, 2.3, 3.3],
            [1.4, 2.4, 3.4],
        ]
    )

    tab = Viewer3DTab()

    # 1. Identity convention: X_SCREEN=+X, Y_SCREEN=+Z (Default)
    model.raw_parameters = {
        "POINT": {
            "X_SCREEN": _wrap(["+X"]),
            "Y_SCREEN": _wrap(["+Z"]),
        }
    }
    tab.update_from_model(model)
    tab.select_all_markers()

    # Check transformed position is unchanged
    np.testing.assert_allclose(
        tab._selected_positions[0], model.markers["Marker1"].position
    )

    # 2. Rotation around X axis: X_SCREEN=+X, Y_SCREEN=+Y
    model.raw_parameters = {
        "POINT": {
            "X_SCREEN": _wrap(["+X"]),
            "Y_SCREEN": _wrap(["+Y"]),
        }
    }
    tab.update_from_model(model)
    # Target transform: [x, y, z] -> [x, -z, y]
    # For [1.0, 2.0, 3.0] -> [1.0, -3.0, 2.0]
    expected = np.array(
        [
            [1.0, -3.0, 2.0],
            [1.1, -3.1, 2.1],
            [1.2, -3.2, 2.2],
            [1.3, -3.3, 2.3],
            [1.4, -3.4, 2.4],
        ]
    )
    np.testing.assert_allclose(tab._selected_positions[0], expected)

    # 3. Custom convention: X_SCREEN=+Y, Y_SCREEN=+Z
    # x_axis=1 (Y), z_axis=2 (Z). Remaining y_axis=0 (X).
    # levi_civita(1, 0, 2) = -1.
    # So y_sign = 1 * 1 * -1 = -1.
    # Target transform: [x, y, z] -> [y, -x, z]
    # For [1.0, 2.0, 3.0] -> [2.0, -1.0, 3.0]
    model.raw_parameters = {
        "POINT": {
            "X_SCREEN": _wrap(["+Y"]),
            "Y_SCREEN": _wrap(["+Z"]),
        }
    }
    tab.update_from_model(model)
    expected_custom = np.array(
        [
            [2.0, -1.0, 3.0],
            [2.1, -1.1, 3.1],
            [2.2, -1.2, 3.2],
            [2.3, -1.3, 3.3],
            [2.4, -1.4, 3.4],
        ]
    )
    np.testing.assert_allclose(tab._selected_positions[0], expected_custom)
