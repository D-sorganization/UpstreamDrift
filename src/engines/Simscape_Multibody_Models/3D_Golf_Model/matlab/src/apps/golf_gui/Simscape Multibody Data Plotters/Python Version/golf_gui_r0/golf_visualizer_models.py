"""Shared data models for the legacy golf swing visualizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameData:
    """Optimized frame data structure"""

    frame_idx: int
    time: float
    # Body points (NumPy arrays for vectorized operations)
    butt: np.ndarray
    clubhead: np.ndarray
    midpoint: np.ndarray
    left_wrist: np.ndarray
    left_elbow: np.ndarray
    left_shoulder: np.ndarray
    right_wrist: np.ndarray
    right_elbow: np.ndarray
    right_shoulder: np.ndarray
    hub: np.ndarray
    # Force/torque vectors for each dataset
    forces: dict[str, np.ndarray]  # 'BASEQ', 'ZTCFQ', 'DELTAQ'
    torques: dict[str, np.ndarray]


@dataclass
class RenderConfig:
    """Complete rendering configuration"""

    # Visibility toggles
    show_forces: dict[str, bool]
    show_torques: dict[str, bool]
    show_body_segments: dict[str, bool]
    show_club: bool = True
    show_face_normal: bool = True
    show_ground: bool = True
    show_ball: bool = True
    show_trajectory: bool = True
    # Visual parameters
    vector_scale: float = 1.0
    body_opacity: float = 0.8
    force_opacity: float = 0.9
    lighting_intensity: float = 1.0
    # Animation settings
    motion_blur: bool = False
    trail_length: int = 30
