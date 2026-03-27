"""Tests for the Anthropic adapter."""

import sys
from unittest.mock import MagicMock

# Mock anthropic globally so lazy imports bypass the missing package
anthropic_mock = MagicMock()
anthropic_mock.OpenAI = MagicMock()
anthropic_mock.Anthropic = MagicMock()
sys.modules["anthropic"] = anthropic_mock
# for gemini
if "anthropic" == "google.generativeai":
    sys.modules["google"] = MagicMock()
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from src.shared.python.ai.adapters.anthropic_adapter import (  # noqa: E402
    AnthropicAdapter,
)
from src.shared.python.ai.adapters.base import ToolDeclaration  # noqa: E402
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
    """Provide a configured AnthropicAdapter."""
    return AnthropicAdapter(api_key="sk-ant", model="claude-3", timeout=30.0)


def test_init(adapter):
    """Test initialization."""
    assert adapter._api_key == "sk-ant"
    assert adapter._model == "claude-3"
    assert adapter._timeout == 30.0
    assert adapter._client is None


def test_get_client(adapter):
    sys.modules["anthropic"].Anthropic.reset_mock()
    """Test client lazy loading."""
    client = adapter._get_client()
    sys.modules["anthropic"].Anthropic.assert_called_once_with(
        api_key="sk-ant", timeout=30.0
    )
    assert adapter._client == client

    sys.modules["anthropic"].Anthropic.reset_mock()
    client2 = adapter._get_client()
    sys.modules["anthropic"].Anthropic.assert_not_called()
    assert client2 == client


def test_get_client_import_error():
    """Test missing anthropic package."""
    adapter = AnthropicAdapter("sk-ant")

    real_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(AIProviderError, match="anthropic package required"),
    ):
        adapter._get_client()


def test_capabilities(adapter):
    """Test capabilities declaration."""
    caps = adapter.capabilities
    assert caps.provider_name == "anthropic"
    assert caps.model_name == "claude-3"
    assert ProviderCapability.FUNCTION_CALLING in caps.supported
    assert ProviderCapability.STREAMING in caps.supported


@patch("src.shared.python.ai.adapters.anthropic_adapter.AnthropicAdapter._get_client")
def test_validate_connection_success(mock_get_client, adapter):
    """Test validate_connection success."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Hi!"
    mock_client.messages.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    success, msg = adapter.validate_connection()
    assert success is True
    assert "Connected to Anthropic with claude-3" in msg

    # Fallback message
    mock_response.content = None
    success, msg = adapter.validate_connection()
    assert success is True
    assert msg == "Connected to Anthropic"


@patch("src.shared.python.ai.adapters.anthropic_adapter.AnthropicAdapter._get_client")
def test_validate_connection_errors(mock_get_client, adapter):
    """Test validate_connection error handling."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.messages.create.side_effect = RuntimeError("authentication failed")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "Invalid API key" in msg

    mock_client.messages.create.side_effect = ValueError("rate limit 429")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "Rate limited" in msg

    mock_client.messages.create.side_effect = OSError("network error")
    success, msg = adapter.validate_connection()
    assert success is False
    assert "network error" in msg


def test_ensure_alternating_roles(adapter):
    """Test merging of consecutive identical roles."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "there"},
        {"role": "assistant", "content": "greetings"},
        {"role": "assistant", "content": [{"type": "text", "text": "friend"}]},
        {"role": "user", "content": [{"type": "text", "text": "hey"}]},
        {"role": "user", "content": "you"},
    ]

    alternated = adapter._ensure_alternating_roles(messages)
    assert len(alternated) == 3

    # 1. user string + string
    assert alternated[0]["role"] == "user"
    assert alternated[0]["content"] == "hi\n\nthere"

    # 2. assistant string + list
    assert alternated[1]["role"] == "assistant"
    assert isinstance(alternated[1]["content"], list)
    assert len(alternated[1]["content"]) == 2
    assert alternated[1]["content"][0]["text"] == "greetings"
    assert alternated[1]["content"][1]["text"] == "friend"

    # 3. user list + string
    assert alternated[2]["role"] == "user"
    assert isinstance(alternated[2]["content"], list)
    assert len(alternated[2]["content"]) == 2
    assert alternated[2]["content"][0]["text"] == "hey"
    assert alternated[2]["content"][1]["text"] == "you"


def test_format_messages(adapter):
    """Test format_messages mapping tool interactions."""
    ctx = ConversationContext()
    ctx.user_expertise = ExpertiseLevel.EXPERT

    tc = MagicMock()
    tc.id = "call_abc"
    tc.name = "get_weather"
    tc.arguments = {"loc": "tokyo"}

    ctx.messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="checking", tool_calls=[tc]),
        Message(role="tool", content="sunny", tool_call_id="call_abc"),
    ]

    # Send another user message
    formatted = adapter._format_messages(ctx, "how are you?")

    # 0 = user, 1 = assistant, 2 = tool (mapped to user), 3 = user -> will merge 2 and 3
    assert len(formatted) == 3

    assert formatted[0]["role"] == "user"
    assert formatted[0]["content"] == "hello"

    assert formatted[1]["role"] == "assistant"
    assert isinstance(formatted[1]["content"], list)
    assert formatted[1]["content"][0]["text"] == "checking"
    assert formatted[1]["content"][1]["type"] == "tool_use"
    assert formatted[1]["content"][1]["id"] == "call_abc"

    assert formatted[2]["role"] == "user"
    assert isinstance(formatted[2]["content"], list)
    assert formatted[2]["content"][0]["type"] == "tool_result"
    assert formatted[2]["content"][0]["content"] == "sunny"
    assert formatted[2]["content"][1]["text"] == "how are you?"


@patch("src.shared.python.ai.adapters.anthropic_adapter.AnthropicAdapter._get_client")
def test_send_message_success(mock_get_client, adapter):
    """Test send_message success."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    block = MagicMock()
    block.type = "text"
    block.text = "Hello world"

    mock_response.content = [block]
    mock_response.stop_reason = "end_turn"
    mock_response.usage.input_tokens = 10
    mock_response.usage.output_tokens = 20
    mock_response.model = "claude-3"
    mock_response.id = "msg_123"

    mock_client.messages.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()
    tools = [ToolDeclaration(name="t1", description="desc1")]

    resp = adapter.send_message("hi", ctx, tools)

    assert resp.content == "Hello world"
    assert len(resp.tool_calls) == 0
    assert resp.finish_reason == "end_turn"
    assert resp.usage["input_tokens"] == 10
    assert resp.metadata["model"] == "claude-3"

    mock_client.messages.create.assert_called_once()


@patch("src.shared.python.ai.adapters.anthropic_adapter.AnthropicAdapter._get_client")
def test_send_message_with_tools(mock_get_client, adapter):
    """Test send_message receiving a tool call."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    block1 = MagicMock()
    block1.type = "text"
    block1.text = "Checking weather"

    block2 = MagicMock()
    block2.type = "tool_use"
    block2.id = "tu_123"
    block2.name = "weather"
    block2.input = {"city": "oslo"}

    mock_response.content = [block1, block2]
    mock_response.stop_reason = "tool_use"
    mock_client.messages.create.return_value = mock_response
    mock_get_client.return_value = mock_client

    resp = adapter.send_message("weather", ConversationContext(), [])

    assert resp.content == "Checking weather"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "weather"
    assert resp.tool_calls[0].id == "tu_123"
    assert resp.tool_calls[0].arguments == {"city": "oslo"}


@patch("src.shared.python.ai.adapters.anthropic_adapter.AnthropicAdapter._get_client")
def test_send_message_error_handling(mock_get_client, adapter):
    """Test error handling in send_message."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    ctx = ConversationContext()

    mock_client.messages.create.side_effect = RuntimeError("rate limit 429")
    with pytest.raises(AIRateLimitError):
        adapter.send_message("msg", ctx, [])

    mock_client.messages.create.side_effect = ValueError("timeout event")
    with pytest.raises(AITimeoutError):
        adapter.send_message("msg", ctx, [])

    mock_client.messages.create.side_effect = OSError("network dropped")
    with pytest.raises(AIConnectionError):
        adapter.send_message("msg", ctx, [])

    mock_client.messages.create.side_effect = RuntimeError("other error")
    with pytest.raises(AIProviderError):
        adapter.send_message("msg", ctx, [])


@patch("src.shared.python.ai.adapters.anthropic_adapter.AnthropicAdapter._get_client")
def test_stream_response(mock_get_client, adapter):
    """Test streaming response."""
    mock_client = MagicMock()

    # Create events
    e1 = MagicMock()
    e1.type = "content_block_delta"
    e1.delta = MagicMock()
    e1.delta.text = "Hel"

    e2 = MagicMock()
    e2.type = "content_block_delta"
    e2.delta = MagicMock()
    e2.delta.text = "lo"

    e3 = MagicMock()
    e3.type = "message_stop"

    mock_stream = [e1, e2, e3]

    class MockStreamContext:
        def __enter__(self):
            return mock_stream

        def __exit__(self, *args):
            pass

    mock_client.messages.stream.return_value = MockStreamContext()
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()
    chunks = list(adapter.stream_response("hi", ctx, []))

    assert len(chunks) == 3
    assert chunks[0].content == "Hel"
    assert chunks[0].is_final is False
    assert chunks[1].content == "lo"
    assert chunks[1].is_final is False
    assert chunks[2].content == ""
    assert chunks[2].is_final is True


@patch("src.shared.python.ai.adapters.anthropic_adapter.AnthropicAdapter._get_client")
def test_stream_error_handling(mock_get_client, adapter):
    """Test streaming error handling."""
    mock_client = MagicMock()

    class FailingStreamContext:
        def __enter__(self):
            raise ValueError("Stream failed")

        def __exit__(self, *args):
            pass

    mock_client.messages.stream.return_value = FailingStreamContext()
    mock_get_client.return_value = mock_client

    ctx = ConversationContext()
    with pytest.raises(AIProviderError):
        list(adapter.stream_response("hi", ctx, []))
