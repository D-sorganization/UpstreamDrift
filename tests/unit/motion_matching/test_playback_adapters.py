"""Tests for Playback Adapters and Capability Matrix (MV-04 #10480)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.shared.python.motion_matching.playback import InterpolatedPlaybackState
from src.shared.python.motion_matching.playback_adapters import (
    GepettoPlaybackAdapter,
    MediaVideoAdapter,
    MeshCatPlaybackAdapter,
    QtPlaybackAdapter,
    ReactPlaybackAdapter,
    get_playback_capability_matrix,
)


def test_playback_capability_matrix_completeness() -> None:
    matrix = get_playback_capability_matrix()
    assert "qt" in matrix
    assert "react" in matrix
    assert "meshcat" in matrix
    assert "gepetto" in matrix
    assert "media_video" in matrix

    for info in matrix.values():
        assert "available" in info
        assert "capabilities" in info
        caps = info["capabilities"]
        assert "supports_pose" in caps
        assert "supports_markers" in caps
        assert "supports_forces" in caps
        assert "supports_video_sync" in caps
        assert "supports_audio" in caps
        assert "supports_paused_camera_orbit" in caps


def test_media_video_adapter_offset_and_rate() -> None:
    # Offset of +0.5s (video started 0.5s before swing)
    adapter = MediaVideoAdapter(media_offset_s=0.5)
    caps = adapter.capabilities()
    assert caps.supports_video_sync is True
    assert caps.supports_audio is False
    assert "Audio" in str(caps.audio_disabled_reason)

    assert adapter.media_time_s(0.0) == pytest.approx(0.5)
    assert adapter.media_time_s(1.2) == pytest.approx(1.7)
    assert adapter.media_time_s(-0.8) == pytest.approx(0.0)


def test_react_playback_adapter_payload_serialization() -> None:
    adapter = ReactPlaybackAdapter()
    state = InterpolatedPlaybackState(
        time_s=1.234,
        q=np.array([0.1, 0.2, 0.3]),
        model_markers=np.array([[0.0, 1.0, 2.0]]),
        target_markers=None,
        forces=None,
        lower_index=3,
        fraction=0.45,
        is_solver_state=False,
    )
    payload = adapter.to_payload(state, speed=1.0, duration_s=1.814)
    assert payload["time_s"] == pytest.approx(1.234)
    assert payload["speed"] == 1.0
    assert payload["is_solver_state"] is False
    assert payload["scrub_value"] >= 0
    assert "q" in payload
    assert "model_markers" in payload


def test_native_adapters_availability_and_reasons() -> None:
    meshcat_adapter = MeshCatPlaybackAdapter()
    if not meshcat_adapter.is_available():
        assert meshcat_adapter.unavailable_reason is not None

    gepetto_adapter = GepettoPlaybackAdapter()
    if not gepetto_adapter.is_available():
        assert gepetto_adapter.unavailable_reason is not None


def test_qt_adapter_capabilities() -> None:
    adapter = QtPlaybackAdapter()
    assert adapter.is_available() is True
    caps = adapter.capabilities()
    assert caps.supports_pose is True
    assert caps.supports_markers is True
    assert caps.supports_forces is True
    assert caps.supports_paused_camera_orbit is True
    assert caps.supports_audio is False
