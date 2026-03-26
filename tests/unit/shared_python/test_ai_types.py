from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.ai.types import (
    AgentChunk,
    AgentResponse,
    ConversationContext,
    ExpertiseLevel,
    Message,
    ProviderCapabilities,
    ProviderCapability,
    ToolCall,
    ToolResult,
)


def test_expertise_level_comparison():
    assert ExpertiseLevel.BEGINNER < ExpertiseLevel.INTERMEDIATE
    assert ExpertiseLevel.INTERMEDIATE <= ExpertiseLevel.ADVANCED
    assert ExpertiseLevel.ADVANCED < ExpertiseLevel.EXPERT

    assert not (ExpertiseLevel.INTERMEDIATE < ExpertiseLevel.BEGINNER)

    # Invalid comparisons
    with pytest.raises(TypeError):
        _ = ExpertiseLevel.BEGINNER < 5  # type: ignore
    with pytest.raises(TypeError):
        _ = ExpertiseLevel.BEGINNER <= 5  # type: ignore


def test_provider_capabilities():
    caps = frozenset([ProviderCapability.FUNCTION_CALLING, ProviderCapability.STREAMING])
    provider = ProviderCapabilities(supported=caps, max_tokens=1000, model_name="gpt-4")

    assert provider.has_capability(ProviderCapability.FUNCTION_CALLING) is True
    assert provider.has_capability(ProviderCapability.VISION) is False


def test_message_dict():
    msg = Message(role="user", content="hello", metadata={"a": 1})
    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "hello"
    assert "timestamp" in d
    assert d["metadata"] == {"a": 1}
    assert "tool_calls" not in d
    assert "tool_call_id" not in d

    tc = ToolCall.create("my_tool", {"arg": "val"})
    msg2 = Message(role="assistant", content="doing it", tool_calls=[tc], tool_call_id="tc_123")
    d2 = msg2.to_dict()
    assert len(d2["tool_calls"]) == 1
    assert d2["tool_calls"][0]["name"] == "my_tool"
    assert d2["tool_call_id"] == "tc_123"


def test_tool_call():
    tc = ToolCall.create("test_tool", {"param": 1})
    assert tc.name == "test_tool"
    assert isinstance(tc.id, str)
    assert tc.id.startswith("tc_")

    d = tc.to_dict()
    assert d["id"] == tc.id
    assert d["name"] == "test_tool"
    assert d["arguments"] == {"param": 1}


def test_tool_result():
    res = ToolResult("tc_1", True, result={"ok": 1}, execution_time=0.5)
    d = res.to_dict()
    assert d["tool_call_id"] == "tc_1"
    assert d["success"] is True
    assert d["result"] == {"ok": 1}
    assert d["error"] is None
    assert d["execution_time"] == 0.5


def test_conversation_context():
    ctx = ConversationContext()
    ctx.add_user_message("hello")
    ctx.add_assistant_message("how are you")
    ctx.add_tool_result("tc_123", "result content")

    assert len(ctx.messages) == 3
    assert ctx.messages[0].role == "user"
    assert ctx.messages[1].role == "assistant"
    assert ctx.messages[2].role == "tool"

    recent = ctx.get_recent_messages(2)
    assert len(recent) == 2
    assert recent[0].role == "assistant"

    d = ctx.to_dict()
    assert d["user_expertise"] == "BEGINNER"
    assert len(d["messages"]) == 3

    ctx2 = ConversationContext.from_dict(d)
    assert len(ctx2.messages) == 3
    assert ctx2.user_expertise == ExpertiseLevel.BEGINNER


def test_conversation_context_save_load(tmp_path: Path):
    file_path = tmp_path / "ctx.json"

    ctx = ConversationContext()
    ctx.add_user_message("test save")
    ctx.user_expertise = ExpertiseLevel.ADVANCED
    ctx.save_to_file(file_path)

    assert file_path.exists()

    ctx2 = ConversationContext.load_from_file(file_path)
    assert ctx2.user_expertise == ExpertiseLevel.ADVANCED
    assert len(ctx2.messages) == 1
    assert ctx2.messages[0].content == "test save"


def test_conversation_context_load_missing():
    ctx = ConversationContext.load_from_file(Path("does_not_exist.json"))
    assert len(ctx.messages) == 0


def test_conversation_context_truncation():
    ctx = ConversationContext(_max_tokens=10)
    for _i in range(10):
        ctx.add_user_message("A very long message string here " * 10)

    # Should truncate but keep at least 2 items initially if logic dictates,
    # but the loop ensures we keep enough that it's truncated properly.
    assert len(ctx.messages) < 10


def test_conversation_context_clear():
    ctx = ConversationContext()
    ctx.add_user_message("hi")
    ctx.clear_history()
    assert len(ctx.messages) == 0


def test_agent_response():
    resp = AgentResponse("hi")
    assert not resp.has_tool_calls

    tc = ToolCall.create("tool", {})
    resp2 = AgentResponse("calling", tool_calls=[tc])
    assert resp2.has_tool_calls

    d = resp2.to_dict()
    assert d["content"] == "calling"
    assert len(d["tool_calls"]) == 1


def test_agent_chunk():
    chunk = AgentChunk(content="chunk data", is_final=True)
    assert chunk.content == "chunk data"
    assert chunk.is_final is True


def test_conversation_context_from_dict_defaults():
    # Test from_dict with missing data
    ctx = ConversationContext.from_dict({})
    assert len(ctx.messages) == 0
    assert ctx.user_expertise == ExpertiseLevel.BEGINNER

    # Also test handling badly formatted timestamp in dict
    ctx = ConversationContext.from_dict(
        {"messages": [{"role": "user", "content": "hello", "timestamp": "bad format"}]}
    )
    assert len(ctx.messages) == 1
    assert ctx.messages[0].timestamp is not None
