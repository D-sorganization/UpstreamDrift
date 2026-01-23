import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.shared.python.ai.types import (
    ConversationContext,
    ExpertiseLevel,
)

# Mock modules
mock_openai_module = MagicMock()
mock_anthropic_module = MagicMock()


@pytest.fixture
def mock_openai_adapter():
    with patch.dict(sys.modules, {"openai": mock_openai_module}):
        from shared.python.ai.adapters.openai_adapter import OpenAIAdapter

        yield OpenAIAdapter(api_key="test-key")


@pytest.fixture
def mock_anthropic_adapter():
    with patch.dict(sys.modules, {"anthropic": mock_anthropic_module}):
        from shared.python.ai.adapters.anthropic_adapter import AnthropicAdapter

        yield AnthropicAdapter(api_key="test-key")


@pytest.fixture
def context():
    return ConversationContext(user_expertise=ExpertiseLevel.BEGINNER)


class TestOpenAIAdapter:
    def test_initialization(self, mock_openai_adapter):
        assert mock_openai_adapter._api_key == "test-key"
        assert mock_openai_adapter._client is None

    def test_get_client_lazy_load(self, mock_openai_adapter):
        client = mock_openai_adapter._get_client()
        assert client is not None
        mock_openai_module.OpenAI.assert_called_once()

    def test_validate_connection_success(self, mock_openai_adapter):
        client_mock = mock_openai_adapter._get_client()
        # Mock models list
        model_mock = Mock()
        model_mock.id = "gpt-4-turbo-preview"
        client_mock.models.list.return_value.data = [model_mock]

        success, msg = mock_openai_adapter.validate_connection()
        assert success is True
        assert "Connected" in msg

    def test_send_message_format(self, mock_openai_adapter, context):
        client_mock = mock_openai_adapter._get_client()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hello", tool_calls=None))]
        client_mock.chat.completions.create.return_value = mock_response

        response = mock_openai_adapter.send_message("Hi", context, [])
        assert response.content == "Hello"

        # Verify call args
        call_kwargs = client_mock.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4-turbo-preview"
        assert len(call_kwargs["messages"]) >= 2  # System + User


class TestAnthropicAdapter:
    def test_initialization(self, mock_anthropic_adapter):
        assert mock_anthropic_adapter._api_key == "test-key"

    def test_validate_connection_success(self, mock_anthropic_adapter):
        client_mock = mock_anthropic_adapter._get_client()
        # Mock message create response
        mock_response = Mock()
        mock_response.content = "Response"
        client_mock.messages.create.return_value = mock_response

        success, msg = mock_anthropic_adapter.validate_connection()
        assert success is True

    def test_format_messages_alternating(self, mock_anthropic_adapter, context):
        # Add messages that are sequential same role
        context.add_user_message("msg1")
        # format_messages adds the current message "msg2" at the end

        msgs = mock_anthropic_adapter._format_messages(context, "msg2")

        # Should merge msg1 and msg2 into one user block or ensure proper structure
        # Implementation details: _format_messages processes history then adds current.
        # History: [User(msg1)]
        # Total: [User(msg1), User(msg2)]
        # _ensure_alternating_roles should merge them.

        assert len(msgs) == 1
        assert "msg1" in msgs[0]["content"]
        assert "msg2" in msgs[0]["content"]

    def test_capabilities(self, mock_anthropic_adapter):
        caps = mock_anthropic_adapter.capabilities
        assert caps.provider_name == "anthropic"
        assert caps.max_tokens == 200000
