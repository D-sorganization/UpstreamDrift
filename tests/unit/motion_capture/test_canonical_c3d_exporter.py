"""Unit tests for the canonical-state → C3D exporter (CC-16).

TDD: tests were written first (red → green → refactor).

Architecture invariant asserted by :class:`TestArchitectureOutputOnly`:
``canonical_c3d_exporter`` is OUTPUT-ONLY.  It must never be imported by
simulation-pipeline internals (trace_io, protocol, pose_interchange).
"""

import ast
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.motion_capture.canonical_c3d_exporter import export_markers_to_c3d
from tests.helpers.ezc3d_support import ezc3d_installed, require_ezc3d

_ezc3d_available = ezc3d_installed()


# ---------------------------------------------------------------------------
# DbC precondition tests (no ezc3d required)
# ---------------------------------------------------------------------------


class TestExportPreconditions:
    """Validate DbC preconditions; these run with or without ezc3d."""

    def test_raises_type_error_on_non_array(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(TypeError, match="numpy.ndarray"):
            export_markers_to_c3d([[1, 2, 3]], ["M1"], 100.0, out)

    def test_raises_on_2d_array(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="shape"):
            export_markers_to_c3d(np.zeros((10, 3)), ["M1"], 100.0, out)

    def test_raises_on_wrong_last_dim(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="shape"):
            export_markers_to_c3d(np.zeros((10, 2, 4)), ["M1", "M2"], 100.0, out)

    def test_raises_on_zero_frames(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="frame"):
            export_markers_to_c3d(np.zeros((0, 1, 3)), ["M1"], 100.0, out)

    def test_raises_on_zero_markers(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="marker"):
            export_markers_to_c3d(np.zeros((10, 0, 3)), [], 100.0, out)

    def test_raises_on_mismatched_name_count(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="marker_names"):
            export_markers_to_c3d(np.zeros((10, 2, 3)), ["M1"], 100.0, out)

    def test_raises_on_zero_sample_rate(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="sample_rate"):
            export_markers_to_c3d(np.zeros((10, 1, 3)), ["M1"], 0.0, out)

    def test_raises_on_negative_sample_rate(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="sample_rate"):
            export_markers_to_c3d(np.zeros((10, 1, 3)), ["M1"], -5.0, out)

    def test_raises_when_output_dir_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent" / "out.c3d"
        with pytest.raises(FileNotFoundError):
            export_markers_to_c3d(np.zeros((10, 1, 3)), ["M1"], 100.0, missing)

    @pytest.mark.unit
    def test_raises_on_invalid_up_axis(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="up_axis"):
            export_markers_to_c3d(np.zeros((10, 1, 3)), ["M1"], 100.0, out, up_axis="x")

    @pytest.mark.unit
    def test_raises_on_invalid_units(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ValueError, match="units"):
            export_markers_to_c3d(
                np.zeros((10, 1, 3)), ["M1"], 100.0, out, units="inches"
            )

    @pytest.mark.skipif(
        _ezc3d_available, reason="ezc3d installed; ImportError won't fire"
    )
    def test_raises_import_error_without_ezc3d(self, tmp_path: Path) -> None:
        out = tmp_path / "out.c3d"
        with pytest.raises(ImportError, match="ezc3d"):
            export_markers_to_c3d(np.zeros((10, 1, 3)), ["M1"], 100.0, out)


@pytest.mark.unit
class TestExportLogicWithMock:
    """Test export logic with mocked ezc3d to verify residual and axis handling."""

    def test_nan_markers_export_as_negative_residual_and_homogeneous_ones(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured_c3d: dict[str, Any] = {}

        class DummyC3D(dict):
            def __init__(self) -> None:
                super().__init__(
                    {
                        "parameters": {
                            "POINT": {
                                "RATE": {},
                                "LABELS": {},
                                "UNITS": {},
                                "DESCRIPTIONS": {},
                            }
                        },
                        "data": {"points": None, "meta_points": {}},
                    }
                )

            def write(self, path: str) -> None:
                captured_c3d["data"] = self["data"]
                captured_c3d["parameters"] = self["parameters"]
                Path(path).touch()

        dummy_mod = type("DummyModule", (), {"c3d": DummyC3D})
        monkeypatch.setattr(
            "src.motion_capture.canonical_c3d_exporter.import_ezc3d",
            lambda: dummy_mod,
        )

        # 3 frames, 2 markers. Marker 1 in frame 1 has NaN.
        markers = np.zeros((3, 2, 3), dtype=np.float64)
        markers[1, 1] = [np.nan, 0.0, 0.0]

        out = export_markers_to_c3d(markers, ["M1", "M2"], 100.0, tmp_path / "out.c3d")
        assert out.exists()

        data = captured_c3d["data"]
        # Check homogeneous 4th row is all 1.0
        points = data["points"]
        assert points.shape == (4, 2, 3)
        np.testing.assert_allclose(points[3, :, :], 1.0)

        # Check meta_points residuals
        residuals = data["meta_points"]["residuals"]
        assert residuals.shape == (1, 2, 3)
        assert residuals[0, 1, 1] == -1.0
        assert residuals[0, 0, 0] == 0.0
        assert residuals[0, 0, 1] == 0.0

    def test_y_up_to_z_up_axis_conversion_and_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured_c3d: dict[str, Any] = {}

        class DummyC3D(dict):
            def __init__(self) -> None:
                super().__init__(
                    {
                        "parameters": {
                            "POINT": {
                                "RATE": {},
                                "LABELS": {},
                                "UNITS": {},
                                "DESCRIPTIONS": {},
                            }
                        },
                        "data": {"points": None, "meta_points": {}},
                    }
                )

            def write(self, path: str) -> None:
                captured_c3d["data"] = self["data"]
                captured_c3d["parameters"] = self["parameters"]
                Path(path).touch()

        dummy_mod = type("DummyModule", (), {"c3d": DummyC3D})
        monkeypatch.setattr(
            "src.motion_capture.canonical_c3d_exporter.import_ezc3d",
            lambda: dummy_mod,
        )

        # Y-up coordinate: X=1.0, Y=2.5 (up), Z=-0.5 (depth)
        markers = np.array([[[1.0, 2.5, -0.5]]], dtype=np.float64)  # (1, 1, 3)

        export_markers_to_c3d(markers, ["M1"], 100.0, tmp_path / "out.c3d", up_axis="y")
        points = captured_c3d["data"]["points"]  # (4, 1, 1)

        # Converted to Z-up: X_zup=1.0, Y_zup=0.5, Z_zup=2.5
        np.testing.assert_allclose(points[0, 0, 0], 1.0)
        np.testing.assert_allclose(points[1, 0, 0], 0.5)
        np.testing.assert_allclose(points[2, 0, 0], 2.5)

        # Check axis convention metadata stored
        params = captured_c3d["parameters"]
        descs = params["POINT"]["DESCRIPTIONS"]["value"]
        assert any("Z" in d or "up_axis" in d for d in descs)

    def test_units_mm_scaling(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured_c3d: dict[str, Any] = {}

        class DummyC3D(dict):
            def __init__(self) -> None:
                super().__init__(
                    {
                        "parameters": {
                            "POINT": {
                                "RATE": {},
                                "LABELS": {},
                                "UNITS": {},
                                "DESCRIPTIONS": {},
                            }
                        },
                        "data": {"points": None, "meta_points": {}},
                    }
                )

            def write(self, path: str) -> None:
                captured_c3d["data"] = self["data"]
                captured_c3d["parameters"] = self["parameters"]
                Path(path).touch()

        dummy_mod = type("DummyModule", (), {"c3d": DummyC3D})
        monkeypatch.setattr(
            "src.motion_capture.canonical_c3d_exporter.import_ezc3d",
            lambda: dummy_mod,
        )

        markers = np.array([[[1.2, 3.4, 5.6]]], dtype=np.float64)  # (1, 1, 3) in metres
        export_markers_to_c3d(markers, ["M1"], 100.0, tmp_path / "out.c3d", units="mm")

        points = captured_c3d["data"]["points"]
        np.testing.assert_allclose(points[0, 0, 0], 1200.0)
        np.testing.assert_allclose(points[1, 0, 0], 3400.0)
        np.testing.assert_allclose(points[2, 0, 0], 5600.0)
        assert captured_c3d["parameters"]["POINT"]["UNITS"]["value"] == ["mm"]

    def test_exposed_in_motion_capture_package(self) -> None:
        import src.motion_capture as mc

        assert hasattr(mc, "export_markers_to_c3d")
        assert callable(mc.export_markers_to_c3d)


# ---------------------------------------------------------------------------
# Round-trip tests (require ezc3d)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ezc3d_available, reason="ezc3d not installed")
class TestExportRoundTrip:
    """Round-trip correctness; skipped when ezc3d is not installed."""

    def test_creates_file(self, tmp_path: Path) -> None:
        markers = np.zeros((5, 2, 3))
        out = export_markers_to_c3d(markers, ["A", "B"], 100.0, tmp_path / "out.c3d")
        assert out.exists()

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        markers = np.zeros((5, 1, 3))
        out = export_markers_to_c3d(markers, ["M1"], 100.0, tmp_path / "out.c3d")
        assert out.is_absolute()

    def test_marker_labels_preserved(self, tmp_path: Path) -> None:
        ezc3d = require_ezc3d()

        markers = np.zeros((5, 2, 3))
        out = export_markers_to_c3d(
            markers, ["LHEE", "RHEE"], 100.0, tmp_path / "out.c3d"
        )
        c = ezc3d.c3d(str(out))
        assert c["parameters"]["POINT"]["LABELS"]["value"] == ["LHEE", "RHEE"]

    def test_round_trip_marker_data(self, tmp_path: Path) -> None:
        ezc3d = require_ezc3d()

        rng = np.random.default_rng(42)
        markers = rng.uniform(-1.0, 1.0, (10, 3, 3))
        names = ["A", "B", "C"]
        out = export_markers_to_c3d(markers, names, 200.0, tmp_path / "out.c3d")

        c = ezc3d.c3d(str(out))
        # ezc3d stores (4, n_markers, T); first 3 rows are X, Y, Z
        data_xyz = c["data"]["points"][:3, :, :]  # (3, n_markers, T)
        recovered = np.transpose(data_xyz, (2, 1, 0))  # → (T, n_markers, 3)
        np.testing.assert_allclose(recovered, markers, rtol=1e-5)

    def test_units_written_as_metres(self, tmp_path: Path) -> None:
        ezc3d = require_ezc3d()

        markers = np.ones((3, 1, 3))
        out = export_markers_to_c3d(markers, ["X"], 100.0, tmp_path / "out.c3d")
        c = ezc3d.c3d(str(out))
        assert c["parameters"]["POINT"]["UNITS"]["value"] == ["m"]

    def test_sample_rate_written(self, tmp_path: Path) -> None:
        ezc3d = require_ezc3d()

        markers = np.zeros((4, 1, 3))
        out = export_markers_to_c3d(markers, ["X"], 250.0, tmp_path / "out.c3d")
        c = ezc3d.c3d(str(out))
        rate = c["parameters"]["POINT"]["RATE"]["value"][0]
        assert abs(rate - 250.0) < 1e-6


# ---------------------------------------------------------------------------
# Architecture test — output-only invariant
# ---------------------------------------------------------------------------


class TestArchitectureOutputOnly:
    """The exporter must never be imported by simulation-pipeline internals.

    c3d is a terminal output format for external biomechanical tools.
    Importing ``canonical_c3d_exporter`` from the trace/pose stack would
    violate the output-only contract.

    Implementation note: we scan source files directly (not via importlib)
    to avoid triggering transitive import chains that require optional
    heavy dependencies (pydantic, mujoco, etc.).
    """

    _REPO_ROOT = Path(__file__).parent.parent.parent.parent

    # Paths relative to the repo root that must not import canonical_c3d_exporter
    _GUARDED_PATHS = [
        "src/shared/python/simulation_backends/trace_io.py",
        "src/shared/python/simulation_backends/protocol.py",
        "src/shared/python/pose_interchange/canonical.py",
        "src/shared/python/pose_interchange/protocol.py",
    ]

    def _scan_source_for_exporter_import(self, rel_path: str) -> bool:
        """Return True if the source file imports canonical_c3d_exporter."""
        src_path = self._REPO_ROOT / rel_path
        if not src_path.exists():
            return False  # file absent — nothing to guard
        source = src_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(src_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "canonical_c3d_exporter" in node.module:
                    return True
                for alias in node.names:
                    if "canonical_c3d_exporter" in alias.name:
                        return True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "canonical_c3d_exporter" in alias.name:
                        return True
        return False

    def test_trace_io_does_not_import_exporter(self) -> None:
        found = self._scan_source_for_exporter_import(
            "src/shared/python/simulation_backends/trace_io.py"
        )
        assert not found, (
            "trace_io imports canonical_c3d_exporter — "
            "c3d must remain OUTPUT-ONLY and never flow back into the pipeline"
        )

    def test_protocol_does_not_import_exporter(self) -> None:
        found = self._scan_source_for_exporter_import(
            "src/shared/python/simulation_backends/protocol.py"
        )
        assert not found, (
            "protocol imports canonical_c3d_exporter — c3d must remain OUTPUT-ONLY"
        )

    def test_pose_interchange_canonical_does_not_import_exporter(self) -> None:
        found = self._scan_source_for_exporter_import(
            "src/shared/python/pose_interchange/canonical.py"
        )
        assert not found, (
            "pose_interchange.canonical imports canonical_c3d_exporter — "
            "c3d must remain OUTPUT-ONLY"
        )
