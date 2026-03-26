"""Tests for the local Ollama adapter."""

import json
import sys
from unittest.mock import MagicMock

# Mock httpx globally so lazy imports bypass the missing package
httpx_mock = MagicMock()
httpx_mock.OpenAI = MagicMock()
httpx_mock.Anthropic = MagicMock()
sys.modules["httpx"] = httpx_mock


class MockConnectError(Exception):
    pass


class MockTimeoutException(Exception):
    pass


httpx_mock.ConnectError = MockConnectError
httpx_mock.TimeoutException = MockTimeoutException

# for gemini
if "httpx" == "google.generativeai":
    sys.modules["google"] = MagicMock()
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def reset_mocks():
    sys.modules["httpx"].reset_mock()


from src.shared.python.ai.adapters.base import ToolDeclaration  # noqa: E402
from src.shared.python.ai.adapters.ollama_adapter import OllamaAdapter  # noqa: E402
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
    AITimeoutError,
)
from src.shared.python.ai.types import (  # noqa: E402
    ConversationContext,
    ExpertiseLevel,
    Message,
    ProviderCapability,
)


@pytest.fixture
def adapter():
    """Provide a configured OllamaAdapter."""
    return OllamaAdapter(host="http://localhost:11434", model="llama3.1:8b", timeout=10.0)


def test_init_defaults():
    """Test defaults when omitted."""
    adapter = OllamaAdapter()
    assert adapter._host == "http://localhost:11434"
    assert adapter._timeout == 120.0


def test_get_client_import_error():
    """Test missing httpx package."""
    adapter = OllamaAdapter()

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("No httpx provided")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(AIProviderError, match="httpx package required"),
    ):
        adapter._get_client()


def test_get_client(adapter):
    sys.modules["httpx"].Client.reset_mock()
    """Test client lazy loading."""
    client = adapter._get_client()
    sys.modules["httpx"].Client.assert_called_once_with(timeout=10.0)
    assert adapter._client == client


def test_capabilities(adapter):
    """Test capabilities declaration handling dynamic models."""
    caps = adapter.capabilities
    assert caps.provider_name == "ollama"
    assert "llama3" in caps.model_name
    assert ProviderCapability.STREAMING in caps.supported
    assert ProviderCapability.FUNCTION_CALLING in caps.supported  # Because llama3 is in the name

    adapter2 = OllamaAdapter(model="deepseek-coder:33b")
    caps2 = adapter2.capabilities
    assert ProviderCapability.FUNCTION_CALLING not in caps2.supported


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_validate_connection_success(mock_get_client, adapter):
    """Test successful connection validation."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "llama3.1:8b"}, {"name": "mistral:latest"}]
    }
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    success, msg = adapter.validate_connection()
    assert success is True
    assert "Connected to Ollama" in msg


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_validate_connection_missing_model(mock_get_client, adapter):
    """Test connection validation where model is missing."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"models": [{"name": "mistral:latest"}]}
    mock_client.get.return_value = mock_response
    mock_get_client.return_value = mock_client

    success, msg = adapter.validate_connection()
    assert success is False
    assert "Model 'llama3.1:8b' not found" in msg


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_validate_connection_errors(mock_get_client, adapter):
    import sys

    sys.modules["httpx"]
    """Test connection validation error handling."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Connection error
    mock_client.get.side_effect = sys.modules["httpx"].ConnectError("refused")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "Cannot connect to Ollama" in msg

    # 500 error
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_client.get.return_value = mock_response
    mock_client.get.side_effect = None
    success, msg = adapter.validate_connection()
    assert success is False
    assert "status 500" in msg


def test_format_messages(adapter):
    """Test conversion of context history."""
    ctx = ConversationContext()
    ctx.user_expertise = ExpertiseLevel.BEGINNER
    ctx.messages = [
        Message(role="user", content="hello"),
        Message(role="tool", content="tool output"),
    ]

    tools = [ToolDeclaration(name="t1", description="d1")]
    formatted = adapter._format_messages(ctx, "how are you?", tools)

    assert len(formatted) == 4
    assert formatted[0]["role"] == "system"
    assert "t1: d1" in formatted[0]["content"]
    assert formatted[1]["role"] == "user"
    assert formatted[2]["role"] == "assistant"  # "tool" is mapped to "assistant"
    assert formatted[3]["role"] == "user"
    assert formatted[3]["content"] == "how are you?"


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_send_message_success(mock_get_client, adapter):
    """Test successful send_message call."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "model": "llama3.1:8b",
        "message": {"content": "Hello!"},
        "done": True,
        "prompt_eval_count": 50,
        "eval_count": 20,
    }
    mock_client.post.return_value = mock_response
    mock_get_client.return_value = mock_client

    resp = adapter.send_message("hi", ConversationContext(), [])

    assert resp.content == "Hello!"
    assert resp.usage["prompt_tokens"] == 50
    assert resp.usage["completion_tokens"] == 20
    assert resp.finish_reason == "stop"


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_send_message_with_tools(mock_get_client, adapter):
    """Test tool parsing in send_message."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {
            "content": "Let me check...",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"c": "oslo"}}}],
        },
    }
    mock_client.post.return_value = mock_response
    mock_get_client.return_value = mock_client

    resp = adapter.send_message("weather", ConversationContext(), [])

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"c": "oslo"}


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_send_message_errors(mock_get_client, adapter):
    import sys

    sys.modules["httpx"]
    """Test exception mapping in send_message."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.post.side_effect = sys.modules["httpx"].ConnectError("broken")
    with pytest.raises(AIConnectionError):
        adapter.send_message("hi", ConversationContext(), [])

    mock_client.post.side_effect = sys.modules["httpx"].TimeoutException("timeout")
    with pytest.raises(AITimeoutError):
        adapter.send_message("hi", ConversationContext(), [])

    mock_client.post.side_effect = ValueError("other error")
    with pytest.raises(AIProviderError):
        adapter.send_message("hi", ConversationContext(), [])


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_stream_response(mock_get_client, adapter):
    """Test streaming chunk iterator handles NDJSON."""
    mock_client = MagicMock()

    class MockStreamContext:
        def __enter__(self):
            class MockResponse:
                def raise_for_status(self):
                    pass

                def iter_lines(self):
                    yield json.dumps({"message": {"content": "Hel"}})
                    yield json.dumps({"message": {"content": "lo"}, "done": True})

            return MockResponse()

        def __exit__(self, *args):
            pass

    mock_client.stream.return_value = MockStreamContext()
    mock_get_client.return_value = mock_client

    chunks = list(adapter.stream_response("test", ConversationContext(), []))

    assert len(chunks) == 2
    assert chunks[0].content == "Hel"
    assert chunks[0].is_final is False
    assert chunks[1].content == "lo"
    assert chunks[1].is_final is True


@patch("src.shared.python.ai.adapters.ollama_adapter.OllamaAdapter._get_client")
def test_list_and_pull(mock_get_client, adapter):
    """Test utility methods for querying and pulling models."""
    mock_client = MagicMock()

    # list_available_models
    resp_tags = MagicMock()
    resp_tags.json.return_value = {"models": [{"name": "m1"}, {"name": "m2"}]}
    mock_client.get.return_value = resp_tags
    mock_get_client.return_value = mock_client

    models = adapter.list_available_models()
    assert models == ["m1", "m2"]

    # pull_model
    with patch("httpx.Client") as mock_dl_client:
        mock_dl_inst = mock_dl_client.return_value.__enter__.return_value
        success = adapter.pull_model("m3")
        assert success is True
        mock_dl_inst.post.assert_called_once_with(
            "http://localhost:11434/api/pull", json={"name": "m3"}
        )
