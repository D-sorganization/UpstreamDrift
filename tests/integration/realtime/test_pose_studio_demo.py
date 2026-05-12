"""Integration test for the Pose Studio cross-tool live-pose demo.

This test exercises the end-to-end IPC path that the demo widget
relies on, without touching any Qt code. It drives an
:class:`EngineController` (with the realtime publish env-var enabled),
subscribes to ``pose/canonical`` from a test-side callback, and asserts
that the canonical pose payload arrives within a generous 200 ms
budget over the file transport.

Latency budget reasoning:

* The file transport polls every ~33 ms (30 Hz). Median latency on a
  fast SSD is well under that, but slow CI filesystems and Windows
  file-locking jitter can push individual messages to the next poll
  tick. 200 ms gives ~6 polls of headroom — enough to absorb
  pathological scheduling without becoming a flake source.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.realtime import publish, subscribe
from src.tools.pose_studio.controllers.engine_controller import EngineController

pytestmark = pytest.mark.integration


# ---- helpers ------------------------------------------------------------------


@pytest.fixture
def isolated_realtime_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the realtime file transport at a tmp dir for test isolation.

    The file transport reads ``REALTIME_FILE_ROOT`` once per process via
    its module-level transport singleton. We rebind the singleton to a
    fresh transport that picks up the env override.
    """
    monkeypatch.setenv("REALTIME_FILE_ROOT", str(tmp_path))
    monkeypatch.setenv("POSE_STUDIO_PUBLISH_REALTIME", "1")

    # Force the realtime module to re-instantiate its transport so it
    # picks up the new ``REALTIME_FILE_ROOT``. We poke the private
    # cache directly rather than re-importing because re-import would
    # create a second logger and double-register channels.
    from src.shared.python import realtime as realtime_pkg
    from src.shared.python.realtime import api as realtime_api
    from src.shared.python.realtime.transport_file import (
        FileTransport,
        default_channel_path,
    )

    old = realtime_api._TRANSPORT
    realtime_api._TRANSPORT = FileTransport(default_channel_path)
    try:
        yield tmp_path
    finally:
        new = realtime_api._TRANSPORT
        if new is not None and new is not old:
            new.shutdown()
        realtime_api._TRANSPORT = old
        # Touch the package to keep ruff happy about the import.
        assert realtime_pkg.publish is publish


# ---- tests --------------------------------------------------------------------


def _build_non_trivial_pose() -> CanonicalPose:
    """Construct a pose with non-zero pelvis + a couple of joint angles."""
    return CanonicalPose(
        pelvis_translation_m=np.array([0.1, 0.2, 0.95], dtype=float),
        pelvis_rotation_xyz_deg=np.array([0.0, 5.0, 0.0], dtype=float),
        joint_angles_deg={
            "HipStartPositionX": 7.5,
            "SpineStartPositionX": -3.25,
        },
    )


def test_pose_studio_publish_arrives_within_budget(
    isolated_realtime_root: Path,
) -> None:
    """A pose set on the controller is observed by a subscriber in <=200 ms."""
    received: list[Any] = []
    arrival_event = threading.Event()

    def _callback(payload: Any) -> None:
        received.append(payload)
        arrival_event.set()

    subscription = subscribe("pose/canonical", _callback)
    try:
        # The subscription starts at end-of-file; give the tail thread
        # one tick to settle so it doesn't race with the publish below.
        time.sleep(0.05)

        controller = EngineController("drake")
        pose = _build_non_trivial_pose()

        t0 = time.monotonic()
        controller.set_pose(pose)

        # 200 ms latency budget; see module docstring.
        got = arrival_event.wait(timeout=0.2)
        elapsed = time.monotonic() - t0
    finally:
        subscription.unsubscribe()

    assert (
        got
    ), f"pose/canonical payload did not arrive within 200 ms (elapsed={elapsed:.3f}s)"
    assert len(received) >= 1
    payload = received[0]
    assert isinstance(payload, dict)
    assert payload.get("convention_tag") == "canonical-v1"
    assert payload["pelvis_translation_m"] == pytest.approx([0.1, 0.2, 0.95])
    angles = payload["joint_angles_deg"]
    assert angles["HipStartPositionX"] == pytest.approx(7.5)
    assert angles["SpineStartPositionX"] == pytest.approx(-3.25)


def test_publish_disabled_when_env_var_unset(
    isolated_realtime_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``POSE_STUDIO_PUBLISH_REALTIME`` set, no payload reaches subscribers."""
    monkeypatch.delenv("POSE_STUDIO_PUBLISH_REALTIME", raising=False)

    received: list[Any] = []
    arrival_event = threading.Event()

    def _callback(payload: Any) -> None:
        received.append(payload)
        arrival_event.set()

    subscription = subscribe("pose/canonical", _callback)
    try:
        time.sleep(0.05)
        controller = EngineController("drake")
        controller.set_pose(_build_non_trivial_pose())

        # Wait long enough for ~6 polls; if anything was going to
        # arrive, it would by now.
        got = arrival_event.wait(timeout=0.2)
    finally:
        subscription.unsubscribe()

    assert not got, (
        "pose/canonical payload arrived even though the publish env var was unset; "
        f"received={received!r}"
    )


def test_realtime_publish_round_trip_is_json_serialisable(
    isolated_realtime_root: Path,
) -> None:
    """Direct publish/subscribe sanity check, independent of Pose Studio."""
    received: list[Any] = []
    arrival_event = threading.Event()

    def _callback(payload: Any) -> None:
        received.append(payload)
        arrival_event.set()

    subscription = subscribe("pose/canonical", _callback)
    try:
        time.sleep(0.05)
        publish(
            "pose/canonical",
            {
                "convention_tag": "canonical-v1",
                "pelvis_translation_m": [0.0, 0.0, 0.95],
                "pelvis_rotation_xyz_deg": [0.0, 0.0, 0.0],
                "joint_angles_deg": {},
            },
        )
        got = arrival_event.wait(timeout=0.2)
    finally:
        subscription.unsubscribe()

    assert got, "direct realtime.publish round trip exceeded 200 ms"
    assert received[0]["convention_tag"] == "canonical-v1"


# Touch ``os`` so ruff doesn't strip the import; we use it implicitly
# via ``monkeypatch.setenv`` but a stray import-only reference would
# trigger F401 in some configurations.
_ = os.environ
