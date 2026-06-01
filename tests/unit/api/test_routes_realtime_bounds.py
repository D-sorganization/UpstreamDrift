"""Bounds tests for ``POST /realtime/publish`` (#6928).

The publish endpoint fans a single request out to every WebSocket subscriber,
so an unbounded channel name or payload is an amplification vector. These
tests assert the channel ``max_length`` and the serialized-payload-size guard.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.rate_limit import limiter
from src.api.routes import realtime as realtime_module
from src.api.routes.realtime import _MAX_CHANNEL_LENGTH, _MAX_PAYLOAD_BYTES


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(realtime_module.router)
    app.state.limiter = limiter
    return TestClient(app)


def test_publish_accepts_small_payload(client: TestClient) -> None:
    """A normal publish with no subscribers still succeeds (delivered=0)."""
    resp = client.post(
        "/realtime/publish",
        json={"channel": "pose/canonical", "payload": {"ok": True}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["delivered"] == 0


def test_publish_rejects_overlong_channel(client: TestClient) -> None:
    """A channel name beyond ``max_length`` is rejected by validation (422)."""
    resp = client.post(
        "/realtime/publish",
        json={
            "channel": "a/" + "b" * (_MAX_CHANNEL_LENGTH + 10),
            "payload": {},
        },
    )
    assert resp.status_code == 422, resp.text


def test_publish_rejects_oversized_payload(client: TestClient) -> None:
    """A payload whose serialized size exceeds the cap is rejected (413)."""
    big = "x" * (_MAX_PAYLOAD_BYTES + 1024)
    resp = client.post(
        "/realtime/publish",
        json={"channel": "pose/canonical", "payload": {"blob": big}},
    )
    assert resp.status_code == 413, resp.text
