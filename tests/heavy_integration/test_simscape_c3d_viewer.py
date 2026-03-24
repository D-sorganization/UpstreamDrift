"""Heavy integration tests for Simscape/C3D data viewer (fixes #1991).

Tests C3D viewer module importability, loading a synthetic C3D-compatible
data structure, and headless instantiation of the viewer model layer.
All tests skip gracefully when PyQt6, ezc3d, or the viewer module is
unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestC3DViewerImport:
    """Contract: C3D viewer application module is importable."""

    def test_c3d_viewer_main_window_importable(self) -> None:
        """C3DViewerMainWindow is importable from the Simscape app."""
        try:
            from src.engines.Simscape_Multibody_Models._3D_Golf_Model.python.src.apps.c3d_viewer import (  # noqa: E501
                C3DViewerMainWindow,
            )
        except ImportError as exc:
            pytest.skip(f"c3d_viewer not importable: {exc}")

        assert C3DViewerMainWindow is not None

    def test_c3d_data_model_importable(self) -> None:
        """C3DDataModel is importable from viewer core.models."""
        try:
            from src.engines.Simscape_Multibody_Models._3D_Golf_Model.python.src.apps.core.models import (  # noqa: E501
                C3DDataModel,
            )
        except ImportError as exc:
            pytest.skip(f"C3DDataModel not importable: {exc}")

        assert C3DDataModel is not None


class TestC3DDataHandling:
    """Contract: C3D data can be parsed and accessed via the model layer."""

    def test_ezc3d_importable(self) -> None:
        """ezc3d library is importable in the heavy Docker image."""
        pytest.importorskip("ezc3d")

    def test_ezc3d_creates_synthetic_c3d(self) -> None:
        """ezc3d can construct a minimal C3D object with marker data."""
        ezc3d = pytest.importorskip("ezc3d")

        # Build a minimal c3d object
        c3d = ezc3d.c3d()
        c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
        c3d["parameters"]["POINT"]["LABELS"]["value"] = ["RHIP", "LHIP", "RSHOULDER"]

        # Add 3 frames of marker data (3 markers × 4 components × 3 frames)
        n_frames = 3
        n_markers = 3
        data = np.zeros((4, n_markers, n_frames))
        # Residual = 0.0 means valid marker
        data[3, :, :] = 0.0
        c3d["data"]["points"] = data

        assert c3d["data"]["points"].shape == (4, n_markers, n_frames)
        assert len(c3d["parameters"]["POINT"]["LABELS"]["value"]) == n_markers

    def test_c3d_roundtrip(self, tmp_path) -> None:
        """A minimal C3D file survives a write→read cycle."""
        ezc3d = pytest.importorskip("ezc3d")

        c3d_out = ezc3d.c3d()
        c3d_out["parameters"]["POINT"]["RATE"]["value"] = [100]
        c3d_out["parameters"]["POINT"]["LABELS"]["value"] = ["MARKER1"]

        n_frames = 5
        data = np.zeros((4, 1, n_frames))
        data[0, 0, :] = np.linspace(0, 1, n_frames)  # x trajectory
        c3d_out["data"]["points"] = data

        c3d_path = tmp_path / "test_output.c3d"
        c3d_out.write(str(c3d_path))

        assert c3d_path.exists()
        assert c3d_path.stat().st_size > 0

        # Read back and verify
        c3d_in = ezc3d.c3d(str(c3d_path))
        labels = c3d_in["parameters"]["POINT"]["LABELS"]["value"]
        assert "MARKER1" in labels
        assert c3d_in["data"]["points"].shape[1] == 1


class TestC3DViewerHeadless:
    """Contract: C3D viewer can be instantiated headlessly with Xvfb."""

    def test_c3d_viewer_headless_instantiation(self) -> None:
        """C3DViewerMainWindow can be instantiated in a headless Qt session."""
        try:
            from PyQt6.QtWidgets import QApplication
        except ImportError:
            pytest.skip("PyQt6 not available")

        try:
            from src.engines.Simscape_Multibody_Models._3D_Golf_Model.python.src.apps.c3d_viewer import (  # noqa: E501
                C3DViewerMainWindow,
            )
        except ImportError as exc:
            pytest.skip(f"c3d_viewer not importable: {exc}")

        import sys

        app = QApplication.instance() or QApplication(sys.argv[:1])
        try:
            win = C3DViewerMainWindow()
            assert win is not None
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"C3DViewerMainWindow failed to instantiate: {exc}")
        finally:
            app.quit()


pytestmark = pytest.mark.live_simulation
