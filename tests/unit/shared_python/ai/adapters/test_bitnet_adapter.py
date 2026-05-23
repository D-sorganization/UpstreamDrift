"""Tests for the BitNet subprocess adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.ai.adapters.bitnet_adapter import BitnetAdapter
from src.shared.python.ai.exceptions import AIProviderError
from src.shared.python.ai.types import ConversationContext


@pytest.fixture
def adapter() -> BitnetAdapter:
    """Provide a configured BitNet adapter."""
    return BitnetAdapter(model="bitnet-test.gguf", bitnet_root="C:/bitnet")


def test_send_message_runs_llama_cli_with_valid_prompt(adapter: BitnetAdapter) -> None:
    """Valid prompts should still invoke the subprocess normally."""
    completed = MagicMock(stdout="User: hi\nAssistant: hello")

    with patch(
        "src.shared.python.ai.adapters.bitnet_adapter.subprocess.run",
        return_value=completed,
    ) as mock_run:
        response = adapter.send_message("hi", ConversationContext(), [])

    assert response.content == "hello"
    cmd = mock_run.call_args.args[0]
    assert cmd[0].endswith("llama-cli")
    assert cmd[1:4] == ["-m", "bitnet-test.gguf", "-p"]


def test_send_message_rejects_oversize_prompt(adapter: BitnetAdapter) -> None:
    """Oversize prompts should be rejected before spawning llama-cli."""
    message = "x" * (adapter._MAX_PROMPT_BYTES + 1)

    with (
        patch(
            "src.shared.python.ai.adapters.bitnet_adapter.subprocess.run"
        ) as mock_run,
        pytest.raises(AIProviderError, match="maximum size"),
    ):
        adapter.send_message(message, ConversationContext(), [])

    mock_run.assert_not_called()


def test_stream_response_rejects_invalid_utf8_prompt(
    adapter: BitnetAdapter,
) -> None:
    """Streaming rejects surrogate-containing text before subprocess launch."""
    chunks = list(adapter.stream_response("bad\udcff", ConversationContext(), []))

    assert len(chunks) == 1
    assert chunks[0].is_final is True
    assert "valid UTF-8 text" in chunks[0].content


def test_stream_response_does_not_spawn_for_invalid_utf8_prompt(
    adapter: BitnetAdapter,
) -> None:
    """Invalid prompt text should fail before opening a subprocess."""
    with patch(
        "src.shared.python.ai.adapters.bitnet_adapter.subprocess.Popen"
    ) as mock_popen:
        list(adapter.stream_response("bad\udcff", ConversationContext(), []))

    mock_popen.assert_not_called()
