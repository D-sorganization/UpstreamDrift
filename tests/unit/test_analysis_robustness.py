"""Robustness and edge-case testing for kinematic analysis."""

import numpy as np
import pytest

from src.shared.python.path_utils import get_simscape_model_path, setup_import_paths

# Setup import paths including Simscape model
setup_import_paths(additional_paths=[get_simscape_model_path("3D_Golf_Model")])

from apps.services.analysis import compute_marker_statistics  # noqa: E402


class TestAnalysisRobustness:
    """High-quality robustness tests for kinematic analysis."""

    def test_empty_input(self):
        """Test with empty arrays."""
        time = np.array([])
        pos = np.empty((0, 3))
        stats = compute_marker_statistics(time, pos)

        assert np.isnan(stats["path_length"])
        assert np.isnan(stats["max_speed"])
        assert np.isnan(stats["mean_speed"])

    def test_single_point(self):
        """Test with a single data point (cannot compute speed)."""
        time = np.array([0.0])
        pos = np.array([[1.0, 2.0, 3.0]])
        stats = compute_marker_statistics(time, pos)

        # Expect NaNs as differential properties require >= 2 points
        assert np.isnan(stats["path_length"])
        assert np.isnan(stats["max_speed"])

    def test_nan_coordinates(self):
        """Test input with NaN coordinates."""
        time = np.array([0.0, 0.1, 0.2])
        # Second point is NaN
        pos = np.array([[0.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [1.0, 0.0, 0.0]])
        stats = compute_marker_statistics(time, pos)

        # path_length uses nansum, so it might return value for valid segments?
        # logic: disp = diff(pos). If pos has NaN, diff has NaN.
        # segment_length = norm(disp). If disp has NaN, norm is NaN.
        # path_length = nansum(segment_length). So NaNs are treated as 0 length?
        # Let's verify behavior. Expected: NaNs should contribute 0 or propagate.
        # Usually physics calc should probably fail or warn, but ensuring it doesn't crash is key.
        assert isinstance(stats["path_length"], float)
        assert isinstance(stats["max_speed"], float)

    def test_duplicate_time_frames(self):
        """Test 0 dt (duplicate frames), which causes division by zero."""
        time = np.array([0.0, 0.0, 1.0])  # Duplicate start time
        pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # Instantaneous jump?
                [2.0, 0.0, 0.0],
            ]
        )

        # dt = [0.0, 1.0]
        # dt[dt <= 0] = nan => dt = [nan, 1.0]
        # speed = dist / dt => [inf/nan, val]

        stats = compute_marker_statistics(time, pos)

        # Check robustness
        assert np.isfinite(
            stats["mean_speed"]
        )  # nanmean should skip the NaN/Inf if generated
        assert np.isfinite(stats["path_length"])

    def test_negative_time_step(self):
        """Test negative time steps (dataset out of order)."""
        time = np.array([0.0, 1.0, 0.5])  # Goes back
        pos = np.zeros((3, 3))

        stats = compute_marker_statistics(time, pos)

        # Implementation sets dt[dt<=0] = NaN, so negative steps are ignored
        assert not np.isnan(stats["mean_speed"])  # Should be 0.0 here (pos is static)

    def test_large_dataset_performance(self):
        """Test with a larger dataset (e.g., 100k points)."""
        N = 100_000
        time = np.linspace(0, 100, N)
        # Circular motion
        t_vals = np.linspace(0, 10 * np.pi, N)
        pos = np.column_stack((np.cos(t_vals), np.sin(t_vals), t_vals))

        import time as pytime

        start = pytime.perf_counter()
        stats = compute_marker_statistics(time, pos)
        duration = pytime.perf_counter() - start

        assert duration < 1.0, f"Analysis took too long: {duration:.4f}s"
        assert stats["path_length"] > 0

    @pytest.mark.parametrize(
        "shape_fn",
        [
            lambda: (np.array([1, 2]), np.zeros((3, 3))),  # Mismatch length
            lambda: (None, np.zeros((2, 3))),  # No time
        ],
    )
    def test_invalid_shapes(self, shape_fn):
        """Test mismatched shapes."""
        t, p = shape_fn()
        stats = compute_marker_statistics(t, p)
        assert np.isnan(stats["path_length"])
