"""Unit tests for the AIP API route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.aip import router
from src.api.dependencies import get_engine_manager


class MockEngineManager:
    def get_available_engines(self):
        return ["mujoco"]


@pytest.fixture
def mock_engine_manager():
    return MockEngineManager()


@pytest.fixture
def app(mock_engine_manager) -> FastAPI:
    """Create a FastAPI app with the AIP router."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[get_engine_manager] = lambda: mock_engine_manager
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_get_capabilities(client: TestClient) -> None:
    """Test getting AIP capabilities."""
    response = client.get("/aip/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "capabilities" in data
    assert "supported_methods" in data
    assert data["protocol_version"] == "2.0"


def test_list_methods(client: TestClient) -> None:
    """Test listing available RPC methods."""
    response = client.get("/aip/methods")
    assert response.status_code == 200
    data = response.json()
    assert "methods" in data
    assert "namespaces" in data
    assert "total" in data


def test_rpc_single_invalid_request(client: TestClient) -> None:
    """Test RPC endpoint with invalid JSON structure."""
    response = client.post("/aip/rpc", json="not a dictionary")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32600  # Invalid Request


def test_rpc_single_method_not_found(client: TestClient) -> None:
    """Test RPC endpoint with unknown method."""
    payload = {"jsonrpc": "2.0", "method": "unknown.method", "params": {}, "id": 1}
    response = client.post("/aip/rpc", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32601  # Method not found


def test_rpc_batch_empty(client: TestClient) -> None:
    """Test RPC endpoint with empty batch."""
    response = client.post("/aip/rpc", json=[])
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32600


def test_rpc_batch_invalid_items(client: TestClient) -> None:
    """Test RPC endpoint with batch containing invalid items."""
    response = client.post("/aip/rpc", json=[1, 2, 3])
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 3
    for item in data:
        assert "error" in item
        assert item["error"]["code"] == -32600


def test_rpc_parse_error(client: TestClient) -> None:
    """Test RPC endpoint with malformed JSON body."""
    response = client.post(
        "/aip/rpc",
        content="invalid json {",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == -32700  # Parse error


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 notifications (issue #8004)
#
# A notification is a request with no ``id``. The spec requires the server to
# return *nothing*. ``handle_rpc`` used to call
# ``make_response(result=None, request_id=None)``, which trips that function's
# own "result or error must be provided" precondition and escaped as a 500.
# ---------------------------------------------------------------------------


def test_rpc_notification_returns_no_content(client: TestClient) -> None:
    """A single notification must produce an empty 204, never a 500."""
    response = client.post("/aip/rpc", json={"jsonrpc": "2.0", "method": "model.list"})

    assert response.status_code == 204
    assert response.content == b""


def test_rpc_notification_for_unknown_method_is_silent(client: TestClient) -> None:
    """Error paths must stay silent for notifications too."""
    response = client.post(
        "/aip/rpc", json={"jsonrpc": "2.0", "method": "does.not.exist"}
    )

    assert response.status_code == 204
    assert response.content == b""


def test_rpc_notification_with_invalid_params_is_silent(client: TestClient) -> None:
    """Invalid params on a notification must not produce a response body."""
    response = client.post(
        "/aip/rpc",
        json={"jsonrpc": "2.0", "method": "model.list", "params": "not-a-container"},
    )

    assert response.status_code == 204
    assert response.content == b""


def test_rpc_batch_of_only_notifications_returns_no_content(
    client: TestClient,
) -> None:
    """An all-notification batch must return nothing, not a -32600 error."""
    response = client.post(
        "/aip/rpc",
        json=[
            {"jsonrpc": "2.0", "method": "model.list"},
            {"jsonrpc": "2.0", "method": "system.ping"},
        ],
    )

    assert response.status_code == 204
    assert response.content == b""


def test_rpc_mixed_batch_returns_only_id_bearing_responses(
    client: TestClient,
) -> None:
    """Only the entries carrying an ``id`` may appear in the batch response."""
    response = client.post(
        "/aip/rpc",
        json=[
            {"jsonrpc": "2.0", "method": "model.list"},
            {"jsonrpc": "2.0", "method": "model.list", "id": 7},
            {"jsonrpc": "2.0", "method": "system.ping"},
        ],
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == 7


def test_rpc_malformed_request_without_id_still_reports_an_error(
    client: TestClient,
) -> None:
    """A request too malformed to be a notification must still be reported.

    A bad ``jsonrpc`` version cannot be treated as a well-formed notification,
    so the server answers with an ``id: null`` error object as the spec allows.
    """
    response = client.post("/aip/rpc", json={"jsonrpc": "1.0", "method": "model.list"})

    assert response.status_code == 200
    data = response.json()
    assert data["error"]["code"] == -32600
