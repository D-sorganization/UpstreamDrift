"""In-process WebSocket pub-sub tests using FastAPI's TestClient."""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.realtime import router as realtime_router

pytestmark = pytest.mark.unit


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(realtime_router)
    return TestClient(app)


def test_publish_with_no_subscribers_returns_zero_delivered(
    client: TestClient,
) -> None:
    r = client.post(
        "/realtime/publish",
        json={"channel": "pose/canonical", "payload": {"hello": "world"}},
    )
    assert r.status_code == 200
    assert r.json() == {"channel": "pose/canonical", "delivered": 0}


def test_publish_invalid_channel_returns_400(client: TestClient) -> None:
    r = client.post(
        "/realtime/publish",
        json={"channel": "BAD/Name", "payload": {}},
    )
    assert r.status_code == 400


def test_subscribe_invalid_channel_closes_connection(client: TestClient) -> None:
    # Connecting with an invalid channel should result in the server closing
    # immediately (policy violation 1008).
    from starlette.websockets import WebSocketDisconnect

    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect("/realtime/subscribe?channel=BAD/Name") as ws,
    ):
        # The server may close before or after accept; receive forces the
        # disconnect to surface as an exception.
        ws.receive_text()


def test_subscribe_round_trip_under_latency_budget(client: TestClient) -> None:
    payload = {"frame": 7, "values": [1.0, 2.0, 3.0]}

    with client.websocket_connect("/realtime/subscribe?channel=pose/canonical") as ws:
        # Give the server a tick to register the subscriber.
        time.sleep(0.05)

        t0 = time.perf_counter()
        r = client.post(
            "/realtime/publish",
            json={"channel": "pose/canonical", "payload": payload},
        )
        assert r.status_code == 200
        assert r.json()["delivered"] == 1

        msg = ws.receive_json()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert msg == payload
        # Generous bound: in-process round-trip should be well under 50ms,
        # but CI variance can push it higher. Document the budget but allow
        # headroom.
        assert elapsed_ms < 500.0, f"latency {elapsed_ms:.1f}ms exceeded ceiling"


def test_multiple_subscribers_each_receive(client: TestClient) -> None:
    with (
        client.websocket_connect("/realtime/subscribe?channel=target/active") as ws_a,
        client.websocket_connect("/realtime/subscribe?channel=target/active") as ws_b,
    ):
        time.sleep(0.05)
        r = client.post(
            "/realtime/publish",
            json={"channel": "target/active", "payload": {"x": 1}},
        )
        assert r.status_code == 200
        assert r.json()["delivered"] == 2
        assert ws_a.receive_json() == {"x": 1}
        assert ws_b.receive_json() == {"x": 1}


def test_subscriber_only_receives_its_channel(client: TestClient) -> None:
    with client.websocket_connect("/realtime/subscribe?channel=pose/canonical") as ws:
        time.sleep(0.05)
        # Publish on a different channel — should NOT be delivered.
        client.post(
            "/realtime/publish",
            json={"channel": "target/active", "payload": {"x": 1}},
        )
        # Now publish on the right channel.
        client.post(
            "/realtime/publish",
            json={"channel": "pose/canonical", "payload": {"y": 2}},
        )
        assert ws.receive_json() == {"y": 2}
