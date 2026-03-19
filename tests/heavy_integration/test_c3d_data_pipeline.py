"""
Heavy Integration Contracts — C3D Motion Capture Data
=====================================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane.

Contract: The c3d library can read/write C3D motion capture files and
integrates with the project's data_io pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.live_simulation
class TestC3DCore:
    """Contract: c3d library reads and writes C3D files."""

    def test_c3d_import(self) -> None:
        """c3d library is importable."""
        try:
            import c3d
        except ImportError:
            pytest.skip("c3d not installed")

        assert hasattr(c3d, "Writer") or hasattr(c3d, "Reader")

    def test_c3d_write_and_read_roundtrip(self) -> None:
        """c3d can write marker data and read it back."""
        try:
            import c3d
        except ImportError:
            pytest.skip("c3d not installed")

        # Create a writer with 3 markers, 100 frames at 100 Hz
        writer = c3d.Writer()

        n_markers = 3
        n_frames = 100

        # Generate synthetic trajectory (markers moving in circles)
        for i in range(n_frames):
            t = i / 100.0
            points = np.zeros((n_markers, 5), dtype=np.float32)
            for j in range(n_markers):
                points[j, 0] = np.cos(2 * np.pi * t + j * 2 * np.pi / n_markers) * 100
                points[j, 1] = np.sin(2 * np.pi * t + j * 2 * np.pi / n_markers) * 100
                points[j, 2] = 0.0  # Z
                points[j, 3] = 0.0  # residual
                points[j, 4] = 0.0  # camera info
            writer.add_frames([(points, np.array([]))])

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".c3d", delete=False) as f:
            c3d_path = Path(f.name)
            writer.write(f)

        # Read it back
        with open(c3d_path, "rb") as f:
            reader = c3d.Reader(f)

            frames_read = 0
            for _i, points, _analog in reader.read_frames():
                frames_read += 1
                assert points.shape[0] == n_markers

        c3d_path.unlink()
        assert frames_read == n_frames

    def test_c3d_marker_labels(self) -> None:
        """c3d writer supports named markers."""
        try:
            import c3d
        except ImportError:
            pytest.skip("c3d not installed")

        writer = c3d.Writer()

        # Set marker labels
        try:
            writer.set_point_labels(["RWRIST", "LWRIST", "RSHOULD"])
        except AttributeError:
            # Older c3d versions may not have this method
            pytest.skip("c3d version does not support set_point_labels")

        # Add one frame
        points = np.zeros((3, 5), dtype=np.float32)
        writer.add_frames([(points, np.array([]))])

        with tempfile.NamedTemporaryFile(suffix=".c3d", delete=False) as f:
            c3d_path = Path(f.name)
            writer.write(f)

        c3d_path.unlink()


@pytest.mark.live_simulation
class TestC3DProjectIntegration:
    """Contract: Project's C3D I/O pipeline works end-to-end."""

    def test_data_utils_c3d_support(self) -> None:
        """data_io.data_utils supports C3D loading."""
        try:
            import c3d  # noqa: F401
        except ImportError:
            pytest.skip("c3d not installed")

        # Verify the loading function exists
        from src.shared.python.data_io.data_utils import load_c3d_data

        assert callable(load_c3d_data)

    def test_export_c3d_support(self) -> None:
        """data_io.export supports C3D export."""
        try:
            import c3d  # noqa: F401
        except ImportError:
            pytest.skip("c3d not installed")

        from src.shared.python.data_io.export import export_to_c3d

        assert callable(export_to_c3d)


pytestmark = pytest.mark.live_simulation
