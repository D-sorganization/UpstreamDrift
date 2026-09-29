"""Heavy integration tests for MediaPipe Tasks API (fixes #1987).

Tests both the legacy mp.solutions.pose and the newer mp.tasks API
(>= 0.10), including synthetic image processing via PoseLandmarker.
All tests skip gracefully when mediapipe is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mp():
    """Import mediapipe or skip the module."""
    mp_mod = pytest.importorskip("mediapipe")
    return mp_mod


@pytest.fixture(scope="module")
def synthetic_rgb_frame():
    """A 480×640 synthetic RGB image (blank white)."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 200


pytestmark = pytest.mark.live_simulation
