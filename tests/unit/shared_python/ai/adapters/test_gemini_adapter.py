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
    """Force the legacy `configure()` + `GenerativeModel` SDK path.

    `sys.modules["google.generativeai"]` above is a bare `MagicMock`, which
    auto-creates *any* attribute asked of it - including `Client`. The
    adapter's `from google.generativeai import Client` therefore succeeds
    under test and `HAS_GEMINI_CLIENT` comes out True, so every test below
    silently exercised the per-instance Client path while asserting against
    `configure` / `GenerativeModel` mocks that were never touched.

    The real `google-generativeai` package exposes no `Client`, so the legacy
    path is what runs in production. Pinning the flags here makes the tests
    exercise the path they claim to, rather than one the mock invented.
    `test_modern_client_path_is_preferred_when_available` covers the other
    branch explicitly.
    """
    with (
        patch("src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI", True),
        patch("src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI_CLIENT", False),
        patch("src.shared.python.ai.adapters.gemini_adapter._GenaiClient", None),
    ):
        yield


from src.shared.python.ai.adapters.gemini_adapter import GeminiAdapter  # noqa: E402
from src.shared.python.ai.exceptions import (  # noqa: E402
    AIConnectionError,
    AIProviderError,
)
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
    """A transport failure raises a typed error, never model content.

    Issue #3179: returning `AgentResponse(content=f"Error: {e}")` made a failed
    request indistinguishable from a model that answered with the word
    "Error", so callers rendered transport failures into the chat transcript.
    The adapter now raises from the `AIProviderError` hierarchy instead.
    """
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")

    mock_chat = MagicMock()
    mock_chat.send_message.side_effect = OSError("Connection refused")

    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.start_chat.return_value = mock_chat

    with pytest.raises(AIConnectionError) as excinfo:
        adapter.send_message("Greetings", ConversationContext(), [])

    # The raised error carries the provider-level classification; the original
    # transport exception is preserved as its cause rather than being
    # flattened into a string.
    assert excinfo.value.provider == "gemini"
    assert excinfo.value.__cause__ is mock_chat.send_message.side_effect


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

    # Two content chunks plus the terminator. Issue #2763 makes "every stream
    # ends with exactly one is_final=True chunk" a contract, so consumers can
    # detect completion without inspecting the generator.
    assert len(chunks) == 3
    assert chunks[0].content == "Hel"
    assert chunks[1].content == "lo"
    assert [c.is_final for c in chunks] == [False, False, True]
    assert chunks[-1].content == ""


def test_stream_error() -> None:
    """A streaming failure raises a typed error rather than yielding it.

    Same contract as `test_send_message_error` (issue #3179): the consumer of
    the generator observes an `AIProviderError`, consistent with the
    synchronous path, instead of a chunk whose content is an error string.
    """
    sys.modules["google.generativeai"].configure.reset_mock()
    adapter = GeminiAdapter("sk")

    mock_chat = MagicMock()
    mock_chat.send_message.side_effect = RuntimeError("Broken pipe")

    mock_model_inst = sys.modules["google.generativeai"].GenerativeModel.return_value
    mock_model_inst.start_chat.return_value = mock_chat

    with pytest.raises(AIProviderError) as excinfo:
        list(adapter.stream_response("test", ConversationContext(), []))

    assert "Broken pipe" in str(excinfo.value)


@pytest.mark.unit
def test_modern_client_path_is_preferred_when_available() -> None:
    """SDK 0.5+ builds the model through a per-instance `Client`.

    The legacy path calls the module-global `genai.configure(api_key=...)`,
    so two adapters holding different keys clobber each other (issue #2756).
    When the SDK offers a `Client`, the adapter must use it and must not touch
    the global configure at all. The autouse fixture pins the legacy branch,
    so this is the one test that opts back into the modern one.
    """
    fake_client_cls = MagicMock()
    with (
        patch("src.shared.python.ai.adapters.gemini_adapter.HAS_GEMINI_CLIENT", True),
        patch(
            "src.shared.python.ai.adapters.gemini_adapter._GenaiClient",
            fake_client_cls,
        ),
    ):
        sys.modules["google.generativeai"].configure.reset_mock()
        adapter = GeminiAdapter("sk-modern", "gemini-test-model")

    fake_client_cls.assert_called_once_with(api_key="sk-modern")
    fake_client_cls.return_value.models.get.assert_called_once_with("gemini-test-model")
    assert adapter._model is fake_client_cls.return_value.models.get.return_value
    # The global configure footgun must stay untouched on this path.
    sys.modules["google.generativeai"].configure.assert_not_called()
