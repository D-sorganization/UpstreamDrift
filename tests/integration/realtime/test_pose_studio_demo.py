"""Integration test for the Pose Studio cross-tool live-pose demo.

This test exercises the end-to-end IPC path that the demo widget
relies on, without touching any Qt code. It drives an
:class:`EngineController` (with the realtime publish env-var enabled),
subscribes to ``pose/canonical`` from a test-side callback, and asserts
that the canonical pose payload *arrives* over the file transport.

Timing policy:

* Functional correctness tests wait on a generous ceiling
  (``ARRIVAL_TIMEOUT_S``) — they only care that the payload eventually
  arrives, not how fast. A hard sub-second wall-clock budget flakes on
  loaded CI hosts (slow filesystems, Windows file-locking jitter), so
  the round-trip latency is *not* asserted here.
* A strict latency budget is checked separately in
  ``test_pose_studio_publish_latency_budget``, gated behind the
  ``benchmark`` marker so it only runs in dedicated perf lanes.
* Subscriber readiness is established with an explicit publish/ack
  handshake rather than a fixed ``time.sleep`` — the subscription tail
  thread starts at end-of-file, so we publish a throwaway "ready" probe
  and block until the callback sees it before driving the real payload.
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

# Generous arrival ceiling for functional correctness tests. This is *not* a
# latency budget — it only needs to be long enough that a payload which is
# going to arrive has arrived even on a heavily loaded CI host. The strict
# latency budget lives in the benchmark-marked test below.
ARRIVAL_TIMEOUT_S = 5.0

# Strict round-trip budget asserted only in the benchmark lane. The file
# transport polls at ~30 Hz, so ~6 poll ticks of headroom.
LATENCY_BUDGET_S = 0.2

# Upper bound for the readiness handshake (subscriber tail thread settling).
READINESS_TIMEOUT_S = 5.0


# Sentinel payload tag used by the readiness handshake; the real callbacks
# ignore it so it never pollutes the captured payload list.
_READY_PROBE_TAG = "ready-probe"


# ---- helpers ------------------------------------------------------------------


def _await_subscriber_ready(
    ready_event: threading.Event, channel: str = "pose/canonical"
) -> None:
    """Block until the subscriber whose callback sets ``ready_event`` is live.

    The subscription tail thread starts at end-of-file, so a payload
    published immediately after :func:`subscribe` can race the thread's
    first poll. Instead of sleeping a fixed interval, publish throwaway
    ``_READY_PROBE_TAG`` probes until the real callback observes one,
    proving the transport round-trip is live before the real payload is
    sent. Callers must have their callback set ``ready_event`` on a probe
    and otherwise ignore it (see :func:`_make_callback`).
    """
    deadline = time.monotonic() + READINESS_TIMEOUT_S
    while not ready_event.is_set() and time.monotonic() < deadline:
        publish(channel, {"convention_tag": _READY_PROBE_TAG})
        ready_event.wait(timeout=0.05)
    if not ready_event.is_set():
        raise AssertionError(
            f"realtime subscriber did not become ready within {READINESS_TIMEOUT_S:.1f}s"
        )


def _make_callback(
    received: list[Any], arrival_event: threading.Event, ready_event: threading.Event
) -> Any:
    """Build a subscriber callback that separates probes from real payloads.

    ``_READY_PROBE_TAG`` payloads only flip ``ready_event`` (handshake);
    all other payloads are appended to ``received`` and flip
    ``arrival_event``.
    """

    def _callback(payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("convention_tag") == _READY_PROBE_TAG
        ):
            ready_event.set()
            return
        received.append(payload)
        arrival_event.set()

    return _callback


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


def test_pose_studio_publish_arrives(
    isolated_realtime_root: Path,
) -> None:
    """A pose set on the controller is eventually observed by a subscriber."""
    received: list[Any] = []
    arrival_event = threading.Event()
    ready_event = threading.Event()

    subscription = subscribe(
        "pose/canonical", _make_callback(received, arrival_event, ready_event)
    )
    try:
        _await_subscriber_ready(ready_event)

        controller = EngineController("drake")
        pose = _build_non_trivial_pose()
        controller.set_pose(pose)

        got = arrival_event.wait(timeout=ARRIVAL_TIMEOUT_S)
    finally:
        subscription.unsubscribe()

    assert (
        got
    ), f"pose/canonical payload did not arrive within {ARRIVAL_TIMEOUT_S:.1f}s ceiling"
    assert len(received) >= 1
    payload = received[0]
    assert isinstance(payload, dict)
    assert payload.get("convention_tag") == "canonical-v1"
    assert payload["pelvis_translation_m"] == pytest.approx([0.1, 0.2, 0.95])
    angles = payload["joint_angles_deg"]
    assert angles["HipStartPositionX"] == pytest.approx(7.5)
    assert angles["SpineStartPositionX"] == pytest.approx(-3.25)


@pytest.mark.benchmark
def test_pose_studio_publish_latency_budget(
    isolated_realtime_root: Path,
) -> None:
    """Strict round-trip latency budget — perf lane only (``benchmark`` marker)."""
    received: list[Any] = []
    arrival_event = threading.Event()
    ready_event = threading.Event()

    subscription = subscribe(
        "pose/canonical", _make_callback(received, arrival_event, ready_event)
    )
    try:
        _await_subscriber_ready(ready_event)

        controller = EngineController("drake")
        pose = _build_non_trivial_pose()

        t0 = time.monotonic()
        controller.set_pose(pose)
        got = arrival_event.wait(timeout=LATENCY_BUDGET_S)
        elapsed = time.monotonic() - t0
    finally:
        subscription.unsubscribe()

    assert got, (
        f"pose/canonical payload did not arrive within {LATENCY_BUDGET_S * 1e3:.0f} ms "
        f"(elapsed={elapsed:.3f}s)"
    )


def test_publish_disabled_when_env_var_unset(
    isolated_realtime_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``POSE_STUDIO_PUBLISH_REALTIME`` set, no payload reaches subscribers."""
    received: list[Any] = []
    arrival_event = threading.Event()
    ready_event = threading.Event()

    subscription = subscribe(
        "pose/canonical", _make_callback(received, arrival_event, ready_event)
    )
    try:
        # Establish readiness *before* disabling the publish env var so the
        # handshake itself is unaffected by the behaviour under test.
        _await_subscriber_ready(ready_event)
        monkeypatch.delenv("POSE_STUDIO_PUBLISH_REALTIME", raising=False)

        controller = EngineController("drake")
        controller.set_pose(_build_non_trivial_pose())

        # The pose must NOT arrive; a short negative-wait is acceptable here
        # because we are proving absence, not asserting a latency budget.
        got = arrival_event.wait(timeout=LATENCY_BUDGET_S)
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
    ready_event = threading.Event()

    subscription = subscribe(
        "pose/canonical", _make_callback(received, arrival_event, ready_event)
    )
    try:
        _await_subscriber_ready(ready_event)
        publish(
            "pose/canonical",
            {
                "convention_tag": "canonical-v1",
                "pelvis_translation_m": [0.0, 0.0, 0.95],
                "pelvis_rotation_xyz_deg": [0.0, 0.0, 0.0],
                "joint_angles_deg": {},
            },
        )
        got = arrival_event.wait(timeout=ARRIVAL_TIMEOUT_S)
    finally:
        subscription.unsubscribe()

    assert got, "direct realtime.publish round trip did not arrive"
    assert received[0]["convention_tag"] == "canonical-v1"


# Touch ``os`` so ruff doesn't strip the import; we use it implicitly
# via ``monkeypatch.setenv`` but a stray import-only reference would
# trigger F401 in some configurations.
_ = os.environ
