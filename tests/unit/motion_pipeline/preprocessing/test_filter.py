"""Unit tests for motion_pipeline.preprocessing.filter."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    KeypointSequence,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.preprocessing.filter import (
    FilterType,
    apply_filter,
)

from ._local_fixtures import (
    make_keypoint_sequence,
    make_marker_trajectory,
    make_sinusoidal_keypoint_sequence,
)


def test_apply_filter_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        apply_filter("not-a-sequence", filter_type=FilterType.BUTTERWORTH)  # type: ignore[arg-type]


def test_apply_filter_keypoints_short_sequence_unchanged() -> None:
    seq = make_keypoint_sequence(num_frames=1, num_kp=2)
    out = apply_filter(seq, filter_type=FilterType.BUTTERWORTH)
    assert out.num_frames == 1


def test_apply_filter_markers_short_trajectory_unchanged() -> None:
    traj = make_marker_trajectory(num_frames=1)
    out = apply_filter(traj, filter_type=FilterType.BUTTERWORTH)
    assert out.num_frames == 1


@pytest.mark.parametrize(
    "filter_type",
    [
        FilterType.BUTTERWORTH,
        FilterType.SAVITZKY_GOLAY,
        FilterType.MEDIAN,
        FilterType.GAUSSIAN,
    ],
)
def test_apply_filter_keypoints_preserves_shape(filter_type: FilterType) -> None:
    seq = make_keypoint_sequence(num_frames=20, num_kp=2)
    out = apply_filter(seq, filter_type=filter_type)
    assert isinstance(out, KeypointSequence)
    assert out.num_frames == seq.num_frames
    assert out.num_keypoints == seq.num_keypoints
    assert out.metadata.get("filtered") is True
    assert out.metadata.get("filter_type") == filter_type.value


def test_butterworth_attenuates_high_frequency_noise() -> None:
    """Low-pass cutoff well below noise frequency should reduce noise power."""
    seq, clean = make_sinusoidal_keypoint_sequence(
        num_frames=200, fps=100.0, freq_hz=2.0, noise_freq_hz=25.0, noise_amp=0.5
    )
    out = apply_filter(
        seq, filter_type=FilterType.BUTTERWORTH, cutoff=6.0, order=4, fps=100.0
    )
    assert isinstance(out, KeypointSequence)
    raw = np.array([f.keypoints[0].x for f in seq.frames])
    filtered = np.array([f.keypoints[0].x for f in out.frames])
    # Filtered signal should be closer to the clean signal than the raw signal.
    raw_err = np.std(raw - clean)
    filt_err = np.std(filtered - clean)
    assert filt_err < raw_err


def test_savgol_smooths_noisy_signal() -> None:
    seq, clean = make_sinusoidal_keypoint_sequence(
        num_frames=200, fps=100.0, freq_hz=2.0, noise_freq_hz=25.0, noise_amp=0.3
    )
    out = apply_filter(seq, filter_type=FilterType.SAVITZKY_GOLAY, fps=100.0)
    assert isinstance(out, KeypointSequence)
    raw = np.array([f.keypoints[0].x for f in seq.frames])
    filtered = np.array([f.keypoints[0].x for f in out.frames])
    raw_err = np.std(raw - clean)
    filt_err = np.std(filtered - clean)
    assert filt_err <= raw_err  # at worst, no worse


def test_butterworth_cutoff_clamped_below_nyquist() -> None:
    """Requesting cutoff above Nyquist should be silently clamped to 0.99."""
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    # Nyquist = 15Hz. cutoff=200 is far above -> should not raise
    out = apply_filter(
        seq, filter_type=FilterType.BUTTERWORTH, cutoff=200.0, order=2, fps=30.0
    )
    assert out.num_frames == seq.num_frames


def test_apply_filter_kalman_smooths_signal() -> None:
    """KALMAN actively smooths via the RTS (Rauch-Tung-Striebel) smoother.

    The implementation runs a forward Kalman pass followed by a backward RTS
    pass. Because the backward pass uses future measurements to refine ALL
    estimates (including sample 0), filtered[0] is NOT required to equal
    raw[0] — that was an artefact of the old causal-only forward filter.
    """
    seq = make_keypoint_sequence(num_frames=15, num_kp=1)
    out = apply_filter(seq, filter_type=FilterType.KALMAN)
    assert isinstance(out, KeypointSequence)
    raw = np.array([f.keypoints[0].x for f in seq.frames])
    filtered = np.array([f.keypoints[0].x for f in out.frames])
    # RTS smoother modifies ALL samples (including index 0) — output differs from input.
    assert not np.allclose(filtered, raw), "Kalman RTS smoother must change the signal"
    assert out.num_frames == seq.num_frames


def test_kalman_filter_converges_on_linear_gaussian_signal() -> None:
    seq, clean = make_sinusoidal_keypoint_sequence(
        num_frames=200, fps=100.0, freq_hz=2.0, noise_freq_hz=25.0, noise_amp=0.5
    )
    out = apply_filter(seq, filter_type=FilterType.KALMAN, fps=100.0)
    assert isinstance(out, KeypointSequence)
    raw = np.array([f.keypoints[0].x for f in seq.frames])
    filtered = np.array([f.keypoints[0].x for f in out.frames])
    raw_err = np.std(raw - clean)
    filt_err = np.std(filtered - clean)
    assert filt_err < raw_err  # currently fails because KALMAN is a no-op


def test_apply_filter_marker_trajectory_preserves_frame_count() -> None:
    traj = make_marker_trajectory(num_frames=30)
    out = apply_filter(traj, filter_type=FilterType.BUTTERWORTH, fps=30.0)
    assert isinstance(out, MarkerTrajectory)
    assert out.num_frames == traj.num_frames
    assert out.metadata.get("filtered") is True


def test_pure_python_filter() -> None:
    from src.shared.python.motion_pipeline.preprocessing._filter_pure_python import (
        apply_filter as pure_apply_filter,
        FilterType as PureFilterType,
        _estimate_fps,
        _keypoints_to_array,
        _markers_to_array,
        _ewma,
    )
    from src.shared.python.motion_pipeline.contracts import (
        KeypointFrame,
        KeypointSequence,
        MarkerFrame,
        MarkerTrajectory,
    )

    # 1. Test unsupported data type raises ValueError
    with pytest.raises(ValueError, match="Unsupported"):
        pure_apply_filter("invalid", filter_type=PureFilterType.BUTTERWORTH)  # type: ignore[arg-type]

    # 2. Test sequences with length < 2 are returned unchanged
    seq_short = make_keypoint_sequence(num_frames=1, num_kp=2)
    assert pure_apply_filter(seq_short) is seq_short

    traj_short = make_marker_trajectory(num_frames=1)
    assert pure_apply_filter(traj_short) is traj_short

    # 3. Test estimate fps edge cases
    # Empty / short list
    assert _estimate_fps([]) == 30.0
    # dt <= 0
    from src.shared.python.motion_pipeline.contracts import Keypoint

    kp = Keypoint(x=0.0, y=0.0, z=0.0, confidence=1.0, name="kp")
    f1 = KeypointFrame(
        timestamp=1.0, keypoints=[kp], frame_index=0, schema_name="custom"
    )
    f2 = KeypointFrame(
        timestamp=0.5, keypoints=[kp], frame_index=1, schema_name="custom"
    )
    assert _estimate_fps([f1, f2]) == 30.0

    # 4. Empty frames arrays
    assert _keypoints_to_array([]).size == 0
    assert _markers_to_array([]).size == 0

    # 5. Butterworth, Savitzky-Golay, Median, Gaussian, Kalman filters on keypoints
    seq = make_keypoint_sequence(num_frames=20, num_kp=2)
    for ft in PureFilterType:
        out = pure_apply_filter(seq, filter_type=ft)
        assert isinstance(out, KeypointSequence)
        assert out.num_frames == seq.num_frames
        assert out.metadata.get("filtered") is True

    # 6. Butterworth, Savitzky-Golay, Median, Gaussian, Kalman filters on markers
    traj = make_marker_trajectory(num_frames=20)
    for ft in PureFilterType:
        out = pure_apply_filter(traj, filter_type=ft)
        assert isinstance(out, MarkerTrajectory)
        assert out.num_frames == traj.num_frames
        assert out.metadata.get("filtered") is True

    # 7. EWMA direct test
    assert _ewma(np.array([])).size == 0
    arr = np.array([[[1.0, 2.0, 3.0]]])
    res = _ewma(arr)
    assert res.shape == arr.shape
    assert res[0, 0, 0] == 1.0

    # 8. test cutoff clamped below nyquist in pure python
    # Nyquist = 15Hz. cutoff = 200 is far above
    out_clamp = pure_apply_filter(
        seq, filter_type=PureFilterType.BUTTERWORTH, cutoff=200.0, fps=30.0
    )
    assert out_clamp.num_frames == seq.num_frames

    # 9. Test unknown filter type (hitting else branch)
    # We can pass an invalid filter type string (via type: ignore or cast)
    out_unknown = pure_apply_filter(seq, filter_type="unknown")  # type: ignore[arg-type]
    assert out_unknown.num_frames == seq.num_frames


def test_pure_python_filter_fallbacks() -> None:
    import sys
    from unittest.mock import patch
    import numpy as np
    from src.shared.python.motion_pipeline.preprocessing._filter_pure_python import (
        _butterworth_filter,
        _savgol_filter,
        _median_filter,
        _gaussian_filter,
        _moving_average,
    )

    data = np.random.rand(10, 2, 3)

    # Test _moving_average window < 2
    res_ma_short = _moving_average(data, window=1)
    assert np.array_equal(res_ma_short, data)

    # Patch sys.modules to raise ImportError for scipy dependencies
    with patch.dict(sys.modules, {"scipy.signal": None, "scipy.ndimage": None}):
        # Test individual filters falling back to _moving_average
        res_butter = _butterworth_filter(data, cutoff=5.0, order=2, fps=30.0)
        assert res_butter.shape == data.shape

        res_savgol = _savgol_filter(data, window_length=5, polyorder=2)
        assert res_savgol.shape == data.shape

        res_median = _median_filter(data, kernel_size=3)
        assert res_median.shape == data.shape

        res_gaussian = _gaussian_filter(data, sigma=1.0)
        assert res_gaussian.shape == data.shape
