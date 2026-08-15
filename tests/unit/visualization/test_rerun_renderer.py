"""Unit tests for the Rerun overlay renderer (epic #8390, D1/#8405).

The renderer is exercised against an injected fake SDK so these tests run
without rerun-sdk installed; live `.rrd` export runs under
``requires_rerun`` and skips when the SDK is absent.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.visualization.rerun_renderer import (
    RerunNotAvailableError,
    RerunRenderConfig,
    export_trace_rrd,
    render_overlay_payload,
    require_rerun,
    rerun_available,
)
from src.shared.python.visualization.viewport import ViewportOverlayPayload

pytestmark = pytest.mark.unit


def _payload(*, with_markers: bool = False, with_wrench: bool = False):
    n = 4
    kwargs = {}
    if with_markers:
        kwargs["markers_xyz"] = np.zeros((n, 2, 3))
        kwargs["marker_names"] = ("a", "b")
    if with_wrench:
        kwargs["wrench"] = np.ones((n, 6))
        kwargs["contact_points_xyz"] = np.zeros((n, 1, 3))
    return ViewportOverlayPayload(
        time_s=np.arange(n) * 0.01,
        trajectory_xyz=np.arange(n * 3, dtype=float).reshape(n, 3),
        **kwargs,
    )


class _FakeStream:
    def __init__(self, application_id: str):
        self.application_id = application_id
        self.saved: list[str] = []
        self.spawned = False
        self.logged: list[tuple[str, object, bool]] = []
        self.times: list[float] = []

    def spawn(self) -> None:
        self.spawned = True

    def save(self, path: str) -> None:
        self.saved.append(path)

    def log(self, entity: str, archetype: object, *, static: bool = False) -> None:
        self.logged.append((entity, archetype, static))

    def set_time(self, timeline: str, *, duration: float) -> None:
        self.times.append(duration)


class _FakeRerun:
    """Minimal rerun-module stand-in recording every call."""

    def __init__(self) -> None:
        self.streams: list[_FakeStream] = []
        self.ViewCoordinates = SimpleNamespace(RIGHT_HAND_Z_UP="RIGHT_HAND_Z_UP")

    def RecordingStream(self, application_id: str) -> _FakeStream:  # noqa: N802 - mirrors SDK name
        stream = _FakeStream(application_id)
        self.streams.append(stream)
        return stream

    @staticmethod
    def Points3D(points, labels=None):  # noqa: N802 - mirrors SDK name
        return ("Points3D", np.asarray(points).shape, labels)

    @staticmethod
    def LineStrips3D(strips):  # noqa: N802 - mirrors SDK name
        return ("LineStrips3D", len(strips))

    @staticmethod
    def Arrows3D(*, origins, vectors):  # noqa: N802 - mirrors SDK name
        return ("Arrows3D", np.asarray(origins).shape, np.asarray(vectors).shape)


def test_config_requires_a_sink() -> None:
    with pytest.raises(ValueError, match="sink"):
        RerunRenderConfig(spawn_viewer=False, rrd_path=None)


def test_render_logs_timeline_and_trajectory() -> None:
    fake = _FakeRerun()
    summary = render_overlay_payload(
        _payload(),
        RerunRenderConfig(rrd_path=Path("out.rrd")),
        rr=fake,
    )
    stream = fake.streams[0]
    assert stream.saved == ["out.rrd"]
    assert stream.spawned is False
    assert stream.times == pytest.approx([0.0, 0.01, 0.02, 0.03])
    static_entities = [e for e, _, static in stream.logged if static]
    assert "overlay" in static_entities  # view coordinates
    assert "overlay/trajectory" in static_entities
    assert summary["frames"] == 4
    assert summary["rrd_path"] == "out.rrd"


def test_render_includes_markers_and_wrench_entities() -> None:
    fake = _FakeRerun()
    summary = render_overlay_payload(
        _payload(with_markers=True, with_wrench=True),
        RerunRenderConfig(rrd_path=Path("out.rrd")),
        rr=fake,
    )
    assert "overlay/markers" in summary["entities"]
    assert "overlay/wrench" in summary["entities"]
    logged_entities = {e for e, _, _ in fake.streams[0].logged}
    assert "overlay/markers" in logged_entities
    assert "overlay/wrench" in logged_entities


def test_missing_sdk_raises_with_install_hint() -> None:
    with patch(
        "src.shared.python.visualization.rerun_renderer.rerun_available",
        return_value=False,
    ):
        with pytest.raises(RerunNotAvailableError, match="visualization"):
            require_rerun()
        with pytest.raises(RerunNotAvailableError):
            render_overlay_payload(
                _payload(), RerunRenderConfig(rrd_path=Path("out.rrd"))
            )


def test_rerun_available_is_boolean() -> None:
    assert rerun_available() in (True, False)


def test_export_trace_rrd_via_fake(tmp_path: Path) -> None:
    from src.shared.python.simulation_backends.protocol import Trace

    n = 5
    trace = Trace(
        t=np.arange(n) * 0.01,
        q=np.linspace(0.0, 1.0, n * 3).reshape(n, 3),
        v=np.zeros((n, 3)),
        u=None,
        dt=0.01,
        backend="test",
        meta={},
    )
    fake = _FakeRerun()
    summary = export_trace_rrd(trace, tmp_path / "trace.rrd", rr=fake)
    assert summary["frames"] == n
    assert fake.streams[0].saved == [str(tmp_path / "trace.rrd")]


@pytest.mark.requires_rerun
def test_live_rrd_export(tmp_path: Path) -> None:
    """With rerun-sdk installed, a real .rrd artifact is produced."""
    pytest.importorskip("rerun")
    out = tmp_path / "live.rrd"
    summary = render_overlay_payload(
        _payload(with_markers=True, with_wrench=True),
        RerunRenderConfig(rrd_path=out),
    )
    assert summary["frames"] == 4
    assert out.exists()
    assert out.stat().st_size > 0
