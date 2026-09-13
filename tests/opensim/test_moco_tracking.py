"""Tests for OpenSim Moco dynamic tracking configuration and TRC sanitization (OS-4).

Exercises pure-Python contracts without requiring OpenSim bindings, plus
OpenSim-dependent tests when available.
"""

from pathlib import Path
import tempfile
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
    sanitize_trc_for_horizon,
    _is_finite_float,
)


@pytest.mark.unit
def test_moco_tracking_config_defaults() -> None:
    """Config has sensible defaults for swing tracking."""
    cfg = MocoTrackingConfig()
    cfg.validate()
    assert cfg.horizon_s == 0.85
    assert cfg.t_start_s == 0.0
    assert cfg.duration_s == 0.85
    assert cfg.mesh_interval_s == 0.02
    assert cfg.num_mesh_intervals == 42 or cfg.num_mesh_intervals == 43
    assert cfg.effort_weight == 1e-4
    assert cfg.marker_weight == 1.0


@pytest.mark.unit
def test_moco_tracking_config_validation_errors() -> None:
    """Config enforces DbC parameter preconditions."""
    with pytest.raises(ValueError, match="t_start_s must be >= 0"):
        MocoTrackingConfig(t_start_s=-0.1).validate()

    with pytest.raises(ValueError, match="horizon_s .* must be > t_start_s"):
        MocoTrackingConfig(horizon_s=0.5, t_start_s=0.5).validate()

    with pytest.raises(ValueError, match="mesh_interval_s must be > 0"):
        MocoTrackingConfig(mesh_interval_s=0.0).validate()

    with pytest.raises(ValueError, match="Goal weights must be non-negative"):
        MocoTrackingConfig(effort_weight=-1.0).validate()

    with pytest.raises(ValueError, match="optim_max_iterations must be > 0"):
        MocoTrackingConfig(optim_max_iterations=0).validate()

    with pytest.raises(ValueError, match="Tolerances must be > 0"):
        MocoTrackingConfig(optim_convergence_tolerance=-1e-3).validate()


@pytest.mark.unit
def test_is_finite_float() -> None:
    """Test float finiteness checker."""
    assert _is_finite_float("1.234")
    assert _is_finite_float("-0.567")
    assert _is_finite_float("0.0")
    assert not _is_finite_float("nan")
    assert not _is_finite_float("NaN")
    assert not _is_finite_float("inf")
    assert not _is_finite_float("-inf")
    assert not _is_finite_float("abc")


@pytest.mark.unit
def test_sanitize_trc_for_horizon_prunes_nan_markers() -> None:
    """Test that markers with NaNs inside the observation window are pruned."""
    lines = [
        "PathFileType\t4\t(X/Y/Z)\ttest.trc\n",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        "100.0\t100.0\t3\t3\tm\t100.0\t1\t3\n",
        "Frame#\tTime\tMarkerGood\t\tMarkerDropout\t\tMarkerLateNaN\t\n",
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2\tX3\tY3\tZ3\n",
        "\n",
        "1\t0.00\t0.1\t0.2\t0.3\tnan\tnan\tnan\t1.0\t1.1\t1.2\n",
        "2\t0.01\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\t1.0\t1.1\t1.2\n",
        "3\t0.02\t0.1\t0.2\t0.3\t0.4\t0.5\t0.6\tnan\t1.1\t1.2\n",
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.trc"
        out_path = Path(tmpdir) / "out.trc"
        in_path.write_text("".join(lines), encoding="utf-8")

        retained = sanitize_trc_for_horizon(
            in_path, out_path, t_start=0.0, t_end=0.015, max_missing_ratio=0.0
        )
        assert retained == ["MarkerGood", "MarkerLateNaN"]
        assert out_path.exists()

        out_lines = out_path.read_text(encoding="utf-8").splitlines()
        assert len(out_lines) >= 8
        assert "MarkerGood" in out_lines[3]
        assert "MarkerLateNaN" in out_lines[3]
        assert "MarkerDropout" not in out_lines[3]
