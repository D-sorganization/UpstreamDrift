"""Tests for the Gemini adapter."""

import sys
from unittest.mock import MagicMock

# Mock google.generativeai globally so lazy imports bypass the missing package
mock_genai_pkg = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = mock_genai_pkg
sys.modules["google.generativeai.types"] = MagicMock()

from collections.abc import Generator  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def reset_mocks() -> None:
    sys.modules["google.generativeai"].reset_mock()


@pytest.fixture(autouse=True)
def patch_has_gemini() -> Generator[None, None, None]:
    with (
        patch("src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI", True),
        patch("src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI_CLIENT", False),
    ):
        yield


from src.shared.python.ai.adapters.gemini_adapter import GeminiAdapter  # noqa: E402
from src.shared.python.ai.types import (  # noqa: E402
    ConversationContext,
    Message,
    ProviderCapability,
)


def test_init_missing_package() -> None:
    """Test behavior when the gemini package is missing."""
    with (
        patch("src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI", False),
        pytest.raises(
            ImportError, match="google-generativeai package is not installed"
        ),
    ):
        GeminiAdapter("api-key")


@patch("src.shared.python.ai.adapters.gemini_adapter.genai.configure")
@patch("src.shared.python.ai.adapters.gemini_adapter.GenerativeModel")
def test_init_success(mock_model_cls, mock_configure) -> None:
    adapter = GeminiAdapter("sk-gemini", "gemini-test-model")

    mock_configure.assert_called_once_with(api_key="sk-gemini")
    mock_model_cls.assert_called_once_with("gemini-test-model")

    assert adapter._api_key == "sk-gemini"
    assert adapter._model_name == "gemini-test-model"


def test_gemini_adapter_capabilities() -> None:
    """Test capabilities properly define vision and streaming."""
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk-gemini")  # nosec B106 - test fixture
    caps = adapter.capabilities

    assert caps.provider_name == "google"
    assert "gemini" in caps.model_name
    assert ProviderCapability.STREAMING in caps.supported
    assert ProviderCapability.VISION in caps.supported
    assert ProviderCapability.FUNCTION_CALLING not in caps.supported


def test_gemini_adapter_validate_connection_success() -> None:
    """Test a successful connection validation."""

    # The generative model is returned by the class constructor mock
    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.generate_content.return_value = MagicMock()

    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")
    success, msg = adapter.validate_connection()

    assert success is True
    assert "Connected successfully" in msg
    mock_model_inst.generate_content.assert_called_once_with("Hello")


def test_validate_connection_failure() -> None:
    """Test a failed connection validation."""

    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.generate_content.side_effect = ValueError("Network out")

    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")
    success, msg = adapter.validate_connection()

    assert success is False
    assert "Connection failed" in msg


def test_build_chat_session() -> None:
    """Test history parser for Gemini chat."""

    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")
    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value

    ctx = ConversationContext()
    ctx.messages = [
        Message(role="user", content="msg 1"),
        Message(role="assistant", content="msg 2"),
        Message(
            role="tool", content="msg 3"
        ),  # should map to 'model' internally as fallback
    ]

    adapter._build_chat_session(ctx, "current message")

    mock_model_inst.start_chat.assert_called_once()
    history_arg = mock_model_inst.start_chat.call_args[1]["history"]

    assert len(history_arg) == 3
    assert history_arg[0] == {"role": "user", "parts": ["msg 1"]}
    assert history_arg[1] == {"role": "model", "parts": ["msg 2"]}
    assert history_arg[2] == {"role": "model", "parts": ["msg 3"]}


def test_build_chat_session_empty_message() -> None:
    """Test that build_chat_session handles empty current message by using the last user message as current message."""
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")
    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value

    ctx = ConversationContext()
    ctx.messages = [
        Message(role="user", content="msg 1"),
        Message(role="assistant", content="msg 2"),
        Message(role="user", content="msg 3"),
    ]

    # In chat_service, message="" is passed, but the actual last message is in history.
    chat, current_msg = adapter._build_chat_session(ctx, "")

    mock_model_inst.start_chat.assert_called_once()
    history_arg = mock_model_inst.start_chat.call_args[1]["history"]

    # History should only contain the messages BEFORE the last user message
    assert len(history_arg) == 2
    assert history_arg[0] == {"role": "user", "parts": ["msg 1"]}
    assert history_arg[1] == {"role": "model", "parts": ["msg 2"]}

    # The current message to send should be the popped user message
    assert current_msg == "msg 3"


def test_gemini_adapter_send_message_success() -> None:
    """Test robust send_message path."""
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")

    mock_chat = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello there"
    mock_chat.send_message.return_value = mock_response

    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.start_chat.return_value = mock_chat

    resp = adapter.send_message("Greetings", ConversationContext(), [])

    assert resp.content == "Hello there"
    mock_chat.send_message.assert_called_once_with("Greetings")


def test_send_message_error() -> None:
    """Test send_message error trap."""
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")

    mock_chat = MagicMock()
    mock_chat.send_message.side_effect = OSError("Connection refused")

    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.start_chat.return_value = mock_chat

    resp = adapter.send_message("Greetings", ConversationContext(), [])

    assert "Error: Connection refused" in resp.content


def test_gemini_adapter_stream_response() -> None:
    """Test streaming chunk iterator."""
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")

    mock_chat = MagicMock()

    c1 = MagicMock()
    c1.text = "Hel"
    c2 = MagicMock()
    c2.text = "lo"

    mock_chat.send_message.return_value = [c1, c2]

    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.start_chat.return_value = mock_chat

    chunks = list(adapter.stream_response("test", ConversationContext(), []))

    assert len(chunks) == 3
    assert chunks[0].content == "Hel"
    assert chunks[1].content == "lo"
    assert chunks[2].is_final is True


def test_stream_error() -> None:
    """Test streaming chunk iterator handles generic exceptions safely."""
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")

    mock_chat = MagicMock()
    mock_chat.send_message.side_effect = RuntimeError("Broken pipe")

    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.start_chat.return_value = mock_chat

    chunks = list(adapter.stream_response("test", ConversationContext(), []))

    assert len(chunks) == 1
    assert "Broken pipe" in chunks[0].content
