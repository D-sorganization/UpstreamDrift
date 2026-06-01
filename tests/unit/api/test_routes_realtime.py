"""Unit tests for the realtime route (POST /realtime/publish and WS /realtime/subscribe)."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, WebSocketDisconnect
from fastapi.testclient import TestClient

from src.api.auth.middleware import LocalUser
from src.api.route_registry import ws_compatible_auth_dependency
from src.api.routes.realtime import _registry, router


@pytest.fixture(autouse=True)
def disable_contracts() -> Any:
    from src.shared.python._contracts_level import ContractLevel

    # Mock both level retrieval methods to completely bypass DBC precondition enforcement in tests
    with (
        patch(
            "src.shared.python.core.contracts.level.get_contract_level",
            return_value=ContractLevel.OFF,
        ),
        patch(
            "src.shared.python._contracts_level.get_contract_level",
            return_value=ContractLevel.OFF,
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_auth_disabled() -> Any:
    """Bypass auth checks for public test cases by forcing local/auth-disabled mode."""
    with patch.dict(
        os.environ, {"GOLF_AUTH_DISABLED": "true", "GOLF_SUITE_MODE": "local"}
    ):
        yield


@pytest.fixture(autouse=True)
def clear_registry() -> None:
    """Ensure in-memory registry is cleared between test cases."""
    _registry._subs.clear()


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the realtime router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


# ── POST /realtime/publish tests ──


def test_publish_success_no_subscribers(client: TestClient) -> None:
    """Publish to a valid channel with no active subscribers."""
    response = client.post(
        "/realtime/publish",
        json={"channel": "pose/canonical", "payload": {"data": "test_val"}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["channel"] == "pose/canonical"
    assert data["delivered"] == 0


def test_publish_invalid_channel_format(client: TestClient) -> None:
    """Publish to a channel with an invalid naming convention (400 Bad Request)."""
    response = client.post(
        "/realtime/publish",
        json={"channel": "invalid_name_no_slash", "payload": {}},
    )
    assert response.status_code == 400
    assert "invalid channel name" in response.json()["detail"]


def test_publish_auth_failure(app: FastAPI) -> None:
    """Publish raises 401 when the auth dependency fails."""

    # Override auth dependency to raise 401
    async def mock_auth_fail() -> None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    app.dependency_overrides[ws_compatible_auth_dependency] = mock_auth_fail
    client = TestClient(app)

    response = client.post(
        "/realtime/publish",
        json={"channel": "pose/canonical", "payload": {}},
    )
    assert response.status_code == 401
    # Cleanup overrides
    app.dependency_overrides.clear()


# ── WS /realtime/subscribe tests ──


def test_ws_subscribe_invalid_channel(client: TestClient) -> None:
    """Subscribing to an invalid channel closes the websocket with code 1008."""
    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/realtime/subscribe?channel=invalid_channel"),
    ):
        pass
    assert exc_info.value.code == 1008


@patch("src.api.routes.realtime.resolve_ws_user")
def test_ws_subscribe_auth_failure(mock_resolve: MagicMock, client: TestClient) -> None:
    """Subscribing when resolve_ws_user returns None (failed auth) terminates the connection."""

    async def mock_resolve_fail(websocket: Any) -> None:
        await websocket.close(code=1008)
        return

    mock_resolve.side_effect = mock_resolve_fail

    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/realtime/subscribe?channel=pose/canonical"),
    ):
        pass
    assert exc_info.value.code == 1008


@patch("src.api.routes.realtime.resolve_ws_user")
def test_ws_subscribe_and_publish_success(
    mock_resolve: MagicMock, client: TestClient
) -> None:
    """Successfully connect, subscribe, broadcast a message, and receive it."""
    mock_resolve.return_value = LocalUser()

    with client.websocket_connect("/realtime/subscribe?channel=pose/canonical") as ws:
        # Check that connection is registered
        assert len(_registry._subs.get("pose/canonical", set())) == 1

        # Publish a message via HTTP
        payload = {"value": 123.45}
        response = client.post(
            "/realtime/publish",
            json={"channel": "pose/canonical", "payload": payload},
        )
        assert response.status_code == 200
        assert response.json()["delivered"] == 1

        # Receive broadcast payload from websocket
        received = ws.receive_json()
        assert received == payload


@patch("src.api.routes.realtime.resolve_ws_user")
def test_publish_best_effort_removes_dead_sockets(
    mock_resolve: MagicMock, client: TestClient
) -> None:
    """If websocket.send_json raises an exception, the subscriber is removed."""
    mock_resolve.return_value = LocalUser()

    with client.websocket_connect("/realtime/subscribe?channel=pose/canonical"):
        # Retrieve the registered socket object
        sockets = list(_registry._subs["pose/canonical"])
        assert len(sockets) == 1
        registered_ws = sockets[0]

        # Mock send_json to raise an exception
        async def mock_send_json(data: Any) -> None:
            raise Exception("Websocket connection dead")

        with patch.object(registered_ws, "send_json", side_effect=mock_send_json):
            # Publish should fail to deliver, but complete successfully and clean up the socket
            response = client.post(
                "/realtime/publish",
                json={"channel": "pose/canonical", "payload": {"data": "test"}},
            )
            assert response.status_code == 200
            assert response.json()["delivered"] == 0

            # The dead subscriber should be removed from the registry
            assert "pose/canonical" not in _registry._subs


@pytest.mark.anyio
async def test_registry_remove_non_existent_channel() -> None:
    """Removing a websocket from a non-existent channel is a safe no-op."""
    mock_ws = MagicMock()
    # Call directly. Since registry is in-memory and locked, we can run this safely.
    await _registry.remove("non-existent-channel", mock_ws)
    # The dictionary should remain empty
    assert not _registry._subs


@patch("src.api.routes.realtime.resolve_ws_user")
def test_ws_subscribe_loop_exception(
    mock_resolve: MagicMock, client: TestClient
) -> None:
    """An unhandled exception in the websocket receive loop is caught and cleaned up."""
    mock_resolve.return_value = LocalUser()

    from fastapi import WebSocket

    async def mock_recv_fail() -> str:
        raise Exception("Forced loop failure")

    with (
        patch.object(WebSocket, "receive_text", side_effect=mock_recv_fail),
        client.websocket_connect("/realtime/subscribe?channel=pose/canonical"),
    ):
        # Connecting triggers the background read loop, which immediately raises and exits
        pass

    # Registry must be empty after cleanup in finally block
    assert "pose/canonical" not in _registry._subs


@pytest.mark.anyio
async def test_registry_remove_multiple_subscribers() -> None:
    """Removing one subscriber from a channel with multiple subscribers does not remove the channel."""
    mock_ws1 = MagicMock()
    mock_ws2 = MagicMock()

    await _registry.add("pose/canonical", mock_ws1)
    await _registry.add("pose/canonical", mock_ws2)

    assert len(await _registry.snapshot("pose/canonical")) == 2

    # Remove one subscriber
    await _registry.remove("pose/canonical", mock_ws1)

    # The channel should still exist because mock_ws2 is still subscribed
    snapshot = await _registry.snapshot("pose/canonical")
    assert len(snapshot) == 1
    assert snapshot[0] == mock_ws2
    assert "pose/canonical" in _registry._subs


@patch("src.api.routes.realtime.resolve_ws_user")
def test_ws_subscribe_disconnect_handled(
    mock_resolve: MagicMock, client: TestClient
) -> None:
    """A clean WebSocketDisconnect is caught and handled silently."""
    mock_resolve.return_value = LocalUser()

    from fastapi import WebSocket

    async def mock_recv_disconnect() -> str:
        raise WebSocketDisconnect(code=1000)

    with (
        patch.object(WebSocket, "receive_text", side_effect=mock_recv_disconnect),
        client.websocket_connect("/realtime/subscribe?channel=pose/canonical"),
    ):
        pass

    assert "pose/canonical" not in _registry._subs
