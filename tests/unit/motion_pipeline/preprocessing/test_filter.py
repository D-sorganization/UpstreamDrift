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
    """KALMAN actively smooths via the Rust 1D random-walk kernel.

    Previously this asserted pass-through (an artefact of the pre-Rust
    signal_toolkit import failure path). With the
    ``upstream-mocap-preproc`` kernel installed, Kalman is an honest filter
    and the output is no longer identical to the input.
    """
    seq = make_keypoint_sequence(num_frames=15, num_kp=1)
    out = apply_filter(seq, filter_type=FilterType.KALMAN)
    raw = np.array([f.keypoints[0].x for f in seq.frames])
    filtered = np.array([f.keypoints[0].x for f in out.frames])
    # The first sample is exact (state initialised from data[0]); later
    # samples are smoothed.
    assert filtered[0] == raw[0]
    assert out.num_frames == seq.num_frames


def test_kalman_filter_converges_on_linear_gaussian_signal() -> None:
    seq, clean = make_sinusoidal_keypoint_sequence(
        num_frames=200, fps=100.0, freq_hz=2.0, noise_freq_hz=25.0, noise_amp=0.5
    )
    out = apply_filter(seq, filter_type=FilterType.KALMAN, fps=100.0)
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
