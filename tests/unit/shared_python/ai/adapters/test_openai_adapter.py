"""Tests for the OpenAI adapter."""

import sys
from unittest.mock import MagicMock

# Mock openai globally so lazy imports bypass the missing package
openai_mock = MagicMock()
openai_mock.OpenAI = MagicMock()
openai_mock.Anthropic = MagicMock()
sys.modules["openai"] = openai_mock
# for gemini
if "openai" == "google.generativeai":
    sys.modules["google"] = MagicMock()
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from src.shared.python.ai.adapters.base import ToolDeclaration  # noqa: E402
from src.shared.python.ai.adapters.openai_adapter import OpenAIAdapter  # noqa: E402
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
    AIRateLimitError,
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
    """Provide a configured OpenAIAdapter."""
    return OpenAIAdapter(api_key="sk-test", model="gpt-4-test", timeout=30.0)


def test_init(adapter):
    """Test initialization."""
    assert adapter._api_key == "sk-test"
    assert adapter._model == "gpt-4-test"
    assert adapter._timeout == 30.0
    assert adapter._client is None


def test_get_client(adapter):
    sys.modules["openai"].OpenAI.reset_mock()
    """Test client lazy loading."""
    client = adapter._get_client()
    sys.modules["openai"].OpenAI.assert_called_once_with(
        api_key="sk-test", organization=None, timeout=30.0
    )
    assert adapter._client == client

    # Second call should return cached client
    sys.modules["openai"].OpenAI.reset_mock()
    client2 = adapter._get_client()
    sys.modules["openai"].OpenAI.assert_not_called()
    assert client2 == client


def test_get_client_import_error():
    """Test missing openai package."""
    adapter = OpenAIAdapter("sk-test")
    # Force ImportError when importing OpenAI
    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(AIProviderError, match="openai package required"),
    ):
        adapter._get_client()


def test_capabilities(adapter):
    """Test capabilities declaration."""
    caps = adapter.capabilities
    assert caps.provider_name == "openai"
    assert caps.model_name == "gpt-4-test"
    assert ProviderCapability.FUNCTION_CALLING in caps.supported
    assert ProviderCapability.STREAMING in caps.supported


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_validate_connection_success(mock_get_client, adapter):
    """Test validate_connection success."""
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "gpt-4-test"
    mock_client.models.list.return_value.data = [mock_model]
    mock_get_client.return_value = mock_client

    success, msg = adapter.validate_connection()
    assert success is True
    assert "Connected to OpenAI with gpt-4-test" in msg


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_validate_connection_model_not_found(mock_get_client, adapter):
    """Test validate_connection when model is not in visible list."""
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "gpt-3.5-turbo"
    mock_client.models.list.return_value.data = [mock_model]
    mock_get_client.return_value = mock_client

    success, msg = adapter.validate_connection()
    assert success is True
    assert "not in visible models" in msg


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_validate_connection_errors(mock_get_client, adapter):
    """Test validate_connection error handling."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    # Auth error
    mock_client.models.list.side_effect = RuntimeError("authentication failed")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "Invalid API key" in msg

    # Rate limit error
    mock_client.models.list.side_effect = ValueError("rate limit exceeded")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "Rate limited" in msg

    # Generic error
    mock_client.models.list.side_effect = OSError("network down")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "network down" in msg

    # Provider error (import failure)
    mock_get_client.side_effect = AIProviderError("import failed", provider="openai")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "openai package not installed" in msg


def test_format_messages(adapter):
    """Test formatting messages for OpenAI."""
    ctx = ConversationContext()
    ctx.user_expertise = ExpertiseLevel.EXPERT

    # Mock some tools
    tc = MagicMock()
    tc.id = "call_1"
    tc.name = "get_weather"
    tc.arguments = {"loc": "tokyo"}

    ctx.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi", tool_calls=[tc]),
        Message(role="tool", content="sunny", tool_call_id="call_1"),
    ]

    formatted = adapter._format_messages(ctx, "how are you?")

    assert len(formatted) == 5  # System + 3 history + 1 current
    assert formatted[0]["role"] == "system"
    assert "expert" in formatted[0]["content"]
    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"] == "hello"

    # Assistant with tool call
    assert formatted[2]["role"] == "assistant"
    assert formatted[2]["tool_calls"][0]["id"] == "call_1"
    assert formatted[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert formatted[2]["tool_calls"][0]["function"]["arguments"] == '{"loc": "tokyo"}'

    # Tool result
    assert formatted[3]["role"] == "tool"
    assert formatted[3]["content"] == "sunny"
    assert formatted[3]["tool_call_id"] == "call_1"

    # Current message
    assert formatted[4]["role"] == "user"
    assert formatted[4]["content"] == "how are you?"


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_send_message_success(mock_get_client, adapter):
    """Test send_message success path."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Here is the analysis."
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"

    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    mock_response.model = "gpt-4-test"
    mock_response.id = "req_123"

    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()
    tools = [ToolDeclaration(name="t1", description="desc1")]

    resp = adapter.send_message("analyze", ctx, tools)

    assert resp.content == "Here is the analysis."
    assert len(resp.tool_calls) == 0
    assert resp.finish_reason == "stop"
    assert resp.usage["total_tokens"] == 30
    assert resp.metadata["model"] == "gpt-4-test"

    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args[1]
    assert kwargs["model"] == "gpt-4-test"
    assert kwargs["temperature"] == 0.7
    assert len(kwargs["tools"]) == 1
    assert kwargs["tools"][0]["function"]["name"] == "t1"


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_send_message_with_tool_call(mock_get_client, adapter):
    """Test send_message receiving a tool call."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = None

    mock_tc = MagicMock()
    mock_tc.id = "call_abc"
    mock_tc.function.name = "get_weather"
    mock_tc.function.arguments = '{"location": "paris"}'
    mock_message.tool_calls = [mock_tc]

    mock_choice.message = mock_message
    mock_choice.finish_reason = "tool_calls"

    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_response.model = "gpt-4"
    mock_response.id = "req_123"

    mock_client.chat.completions.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()
    resp = adapter.send_message("weather?", ctx, [])

    assert resp.content == ""  # Null content handled gracefully
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"location": "paris"}
    assert resp.tool_calls[0].id == "call_abc"


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_send_message_error_handling(mock_get_client, adapter):
    """Test error handling in send_message."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()

    # Rate limit
    mock_client.chat.completions.create.side_effect = RuntimeError("rate limit 429")
    with pytest.raises(AIRateLimitError):
        adapter.send_message("msg", ctx, [])

    # Timeout
    mock_client.chat.completions.create.side_effect = ValueError("timeout occurred")
    with pytest.raises(AITimeoutError):
        adapter.send_message("msg", ctx, [])

    # Connection
    mock_client.chat.completions.create.side_effect = OSError("network reset")
    with pytest.raises(AIConnectionError):
        adapter.send_message("msg", ctx, [])

    # Generic
    mock_client.chat.completions.create.side_effect = RuntimeError("unknown issue")
    with pytest.raises(AIProviderError):
        adapter.send_message("msg", ctx, [])


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_stream_response(mock_get_client, adapter):
    """Test streaming response."""
    mock_client = MagicMock()

    # Create chunks
    c1 = MagicMock()
    c1.choices = [MagicMock()]
    c1.choices[0].delta = MagicMock()
    c1.choices[0].delta.content = "Hel"
    c1.choices[0].delta.tool_calls = None
    c1.choices[0].finish_reason = None

    c2 = MagicMock()
    c2.choices = [MagicMock()]
    c2.choices[0].delta = MagicMock()
    c2.choices[0].delta.content = "lo"
    c2.choices[0].delta.tool_calls = None
    c2.choices[0].finish_reason = "stop"

    mock_client.chat.completions.create.return_value = [c1, c2]
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()
    chunks = list(adapter.stream_response("hi", ctx, []))

    assert len(chunks) == 2
    assert chunks[0].content == "Hel"
    assert chunks[0].is_final is False
    assert chunks[0].index == 0

    assert chunks[1].content == "lo"
    assert chunks[1].is_final is True
    assert chunks[1].index == 1


@patch("src.shared.python.ai.adapters.openai_adapter.OpenAIAdapter._get_client")
def test_stream_error_handling(mock_get_client, adapter):
    """Test streaming error propagation."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = ValueError("Stream closed")
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()
    with pytest.raises(AIProviderError):
        # We must consume the iterator to trigger the error
        list(adapter.stream_response("hi", ctx, []))
