"""Tests for the Qt-free native-viewer handoff helper (#11987).

Covers the logic extracted from ``gui.py``'s ``_on_open_native_viewer`` into
``src/tools/matched_swing_browser/native_handoff.py``: NPZ loading (fail
closed on malformed archives, unlike the desktop method it replaces), backend
availability listing, and the launch call forwarded to ``open_in_native_viewer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.native_viewers import (
    ViewerLaunchConfig,
    ViewerLaunchResult,
    ViewerUnavailableError,
)
from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
)
from src.tools.matched_swing_browser import native_handoff

pytestmark = pytest.mark.unit


def _write_npz(path: Path, **arrays: Any) -> None:
    np.savez(path, **arrays)


def test_load_simulation_data_with_coordinates(tmp_path: Path) -> None:
    npz_path = tmp_path / "candidate.npz"
    time_s = np.array([0.0, 0.1, 0.2])
    coordinates = np.zeros((3, 4))
    _write_npz(npz_path, time_s=time_s, coordinates=coordinates, q=np.ones((3, 4)))

    data = native_handoff.load_simulation_data(npz_path)

    assert isinstance(data, SimulationData)
    np.testing.assert_array_equal(data.time_s, time_s)
    np.testing.assert_array_equal(data.q, coordinates)


def test_load_simulation_data_falls_back_to_q(tmp_path: Path) -> None:
    npz_path = tmp_path / "candidate.npz"
    time_s = np.array([0.0, 0.1])
    q = np.ones((2, 3))
    _write_npz(npz_path, time_s=time_s, q=q)

    data = native_handoff.load_simulation_data(npz_path)

    np.testing.assert_array_equal(data.time_s, time_s)
    np.testing.assert_array_equal(data.q, q)


def test_load_simulation_data_missing_file_raises_file_not_found(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        native_handoff.load_simulation_data(tmp_path / "missing.npz")


def test_load_simulation_data_missing_time_s_raises_value_error(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "candidate.npz"
    _write_npz(npz_path, q=np.ones((2, 3)))

    with pytest.raises(ValueError, match="time_s"):
        native_handoff.load_simulation_data(npz_path)


def test_load_simulation_data_missing_coordinates_and_q_raises_value_error(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "candidate.npz"
    _write_npz(npz_path, time_s=np.array([0.0, 0.1]))

    with pytest.raises(ValueError, match="coordinates"):
        native_handoff.load_simulation_data(npz_path)


def test_list_native_viewer_backends_shape_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.shared.python.motion_matching import native_viewers

    for name in native_viewers.get_supported_backends():
        adapter = native_viewers.get_backend_adapter(name)
        monkeypatch.setattr(adapter, "is_available", lambda *, _n=name: _n == "meshcat")

    backends = native_handoff.list_native_viewer_backends()

    assert [b["backend"] for b in backends] == native_viewers.get_supported_backends()
    for entry in backends:
        assert set(entry) == {"backend", "available", "install_hint"}
        assert entry["available"] == (entry["backend"] == "meshcat")
        assert isinstance(entry["install_hint"], str)


def test_launch_native_viewer_passes_data_backend_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz_path = tmp_path / "candidate.npz"
    time_s = np.array([0.0, 0.1])
    q = np.ones((2, 3))
    _write_npz(npz_path, time_s=time_s, q=q)

    captured: dict[str, Any] = {}

    def fake_open_in_native_viewer(
        candidate: SimulationData,
        engine: str,
        config: ViewerLaunchConfig,
        **kwargs: Any,
    ) -> ViewerLaunchResult:
        captured["candidate"] = candidate
        captured["engine"] = engine
        captured["config"] = config
        return ViewerLaunchResult(success=True, backend=engine)

    monkeypatch.setattr(
        native_handoff, "open_in_native_viewer", fake_open_in_native_viewer
    )

    result = native_handoff.launch_native_viewer(npz_path, "mujoco", speed=2.0)

    assert result.success is True
    assert result.backend == "mujoco"
    assert captured["engine"] == "mujoco"
    np.testing.assert_array_equal(captured["candidate"].time_s, time_s)
    np.testing.assert_array_equal(captured["candidate"].q, q)
    assert captured["config"].speed == 2.0
    assert captured["config"].view_mode == "fitted"


def test_launch_native_viewer_propagates_viewer_unavailable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    npz_path = tmp_path / "candidate.npz"
    _write_npz(npz_path, time_s=np.array([0.0]), q=np.ones((1, 1)))

    def fake_open_in_native_viewer(*args: Any, **kwargs: Any) -> ViewerLaunchResult:
        raise ViewerUnavailableError("MuJoCo is not installed")

    monkeypatch.setattr(
        native_handoff, "open_in_native_viewer", fake_open_in_native_viewer
    )

    with pytest.raises(ViewerUnavailableError):
        native_handoff.launch_native_viewer(npz_path, "mujoco")


@pytest.mark.parametrize("backend", [123, None, 1.5])
def test_launch_native_viewer_rejects_non_string_backend(
    tmp_path: Path, backend: Any
) -> None:
    npz_path = tmp_path / "candidate.npz"
    _write_npz(npz_path, time_s=np.array([0.0]), q=np.ones((1, 1)))

    with pytest.raises(TypeError):
        native_handoff.launch_native_viewer(npz_path, backend)


def test_launch_native_viewer_rejects_empty_backend(tmp_path: Path) -> None:
    npz_path = tmp_path / "candidate.npz"
    _write_npz(npz_path, time_s=np.array([0.0]), q=np.ones((1, 1)))

    with pytest.raises(ValueError, match="non-empty"):
        native_handoff.launch_native_viewer(npz_path, "   ")
