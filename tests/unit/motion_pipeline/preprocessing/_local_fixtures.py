"""Self-contained fixtures for preprocessing unit tests.

This file is intentionally local and small to avoid colliding with the
shared ``_fixtures.py`` being introduced by PR #4620. Helpers build CIR
objects (KeypointSequence / MarkerTrajectory) from numpy arrays.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.shared.python.motion_pipeline.contracts import (
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
)


def make_keypoint_sequence(
    num_frames: int = 30,
    num_kp: int = 3,
    fps: float = 30.0,
    confidence: float = 1.0,
    seed: int = 0,
) -> KeypointSequence:
    """Build a KeypointSequence of synthetic 3D keypoints."""
    rng = np.random.default_rng(seed)
    frames: list[KeypointFrame] = []
    for i in range(num_frames):
        kps = [
            Keypoint(
                x=float(rng.normal()),
                y=float(rng.normal()),
                z=float(rng.normal()),
                confidence=confidence,
                name=f"kp_{j}",
            )
            for j in range(num_kp)
        ]
        frames.append(
            KeypointFrame(
                timestamp=i / fps,
                keypoints=kps,
                schema_name="custom",
                frame_index=i,
            )
        )
    return KeypointSequence(id="seq_test", frames=frames)


def make_marker_trajectory(
    num_frames: int = 30,
    marker_names: Iterable[str] = ("RASI", "LASI", "RKNE"),
    fps: float = 30.0,
    seed: int = 0,
) -> MarkerTrajectory:
    """Build a MarkerTrajectory of synthetic 3D markers."""
    rng = np.random.default_rng(seed)
    names = list(marker_names)
    frames: list[MarkerFrame] = []
    for i in range(num_frames):
        markers = {
            name: Marker(
                name=name,
                x=float(rng.normal()),
                y=float(rng.normal() + 1.0),  # positive y bias
                z=float(rng.normal()),
                occluded=False,
            )
            for name in names
        }
        frames.append(MarkerFrame(timestamp=i / fps, markers=markers, frame_index=i))
    return MarkerTrajectory(id="traj_test", frames=frames)


def make_sinusoidal_keypoint_sequence(
    num_frames: int = 200,
    fps: float = 100.0,
    freq_hz: float = 2.0,
    noise_freq_hz: float = 25.0,
    noise_amp: float = 0.2,
    seed: int = 0,
) -> tuple[KeypointSequence, np.ndarray]:
    """Sinusoid + high-frequency noise; useful to test low-pass filters."""
    rng = np.random.default_rng(seed)
    t = np.arange(num_frames) / fps
    signal = np.sin(2 * np.pi * freq_hz * t)
    noise = noise_amp * np.sin(2 * np.pi * noise_freq_hz * t) + 0.01 * rng.standard_normal(
        num_frames
    )
    frames: list[KeypointFrame] = []
    for i, ts in enumerate(t):
        kp = Keypoint(
            x=float(signal[i] + noise[i]),
            y=0.0,
            z=0.0,
            confidence=1.0,
            name="kp",
        )
        frames.append(
            KeypointFrame(
                timestamp=float(ts),
                keypoints=[kp],
                schema_name="custom",
                frame_index=i,
            )
        )
    return KeypointSequence(id="seq_sin", frames=frames), signal


def make_marker_trajectory_with_occlusion(
    num_frames: int = 20,
    occluded_range: tuple[int, int] = (5, 8),
) -> MarkerTrajectory:
    """Marker trajectory with a contiguous occluded window."""
    frames: list[MarkerFrame] = []
    for i in range(num_frames):
        occluded = occluded_range[0] <= i <= occluded_range[1]
        markers = {
            "M1": Marker(
                name="M1",
                x=float(i) * 0.1,
                y=0.0,
                z=0.0,
                occluded=occluded,
            ),
            "M2": Marker(
                name="M2",
                x=0.0,
                y=float(i) * 0.05,
                z=0.0,
                occluded=False,
            ),
        }
        frames.append(MarkerFrame(timestamp=i / 30.0, markers=markers, frame_index=i))
    return MarkerTrajectory(id="traj_occ", frames=frames)


def make_low_confidence_keypoint_sequence(
    num_frames: int = 20,
    low_conf_range: tuple[int, int] = (5, 8),
) -> KeypointSequence:
    """Keypoint sequence with a contiguous low-confidence window."""
    frames: list[KeypointFrame] = []
    for i in range(num_frames):
        conf = 0.1 if low_conf_range[0] <= i <= low_conf_range[1] else 1.0
        kp = Keypoint(x=float(i) * 0.1, y=0.0, z=0.0, confidence=conf, name="kp")
        frames.append(
            KeypointFrame(
                timestamp=i / 30.0,
                keypoints=[kp],
                schema_name="custom",
                frame_index=i,
            )
        )
    return KeypointSequence(id="seq_lowconf", frames=frames)
