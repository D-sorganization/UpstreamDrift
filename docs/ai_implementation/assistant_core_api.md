# Assistant Core API

`src/shared/python/ai/assistant_core.py` provides `AssistantSession`: a
provider-agnostic, headless AI session class with **no dependency on PyQt** or
any other desktop toolkit.  It can be embedded in web dashboards, REST APIs,
WebSocket handlers, CLI tools, or automated tests.

---

## Quick start

```python
import asyncio
from src.shared.python.ai.assistant_core import AssistantSession

session = AssistantSession(provider="anthropic", adapter_kwargs={
    "api_key": "sk-ant-...",
    "model": "claude-3-5-sonnet-20241022",
})

async def main() -> None:
    async for chunk in session.send_message("What engines are available?"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

---

## Injecting custom tool sets

Pass a `ToolRegistry` at construction time.  Any tools registered in it are
made available to the AI on every call to `send_message`.

```python
from src.shared.python.ai.tool_registry import ToolRegistry
from src.shared.python.ai.assistant_core import AssistantSession

registry = ToolRegistry()

@registry.register(name="list_engines", description="List available physics engines")
def list_engines() -> list[str]:
    return ["mujoco", "drake", "pinocchio"]

session = AssistantSession(provider="ollama", tool_registry=registry)
```

The registry's `declarations()` method converts registered tools to
`ToolDeclaration` objects before passing them to the underlying adapter.

### Confirmation-gated tools

Register tools with `requires_confirmation=True` to prevent automatic
execution.  Supply an async `confirmation_callback` to let the calling
application decide:

```python
async def my_confirm(tool_name: str, arguments: dict) -> bool:
    # Could display a prompt, check permissions, etc.
    return tool_name != "delete_file"

session = AssistantSession(
    provider="anthropic",
    tool_registry=registry,
    confirmation_callback=my_confirm,
)
```

If no `confirmation_callback` is provided, gated tools are skipped and a
`[Tool <name>: skipped (confirmation required)]` notice is injected into
the stream.

---

## Injecting custom context providers (RAG)

Pass any object whose `query(text: str, top_k: int) -> list[tuple[doc, float]]`
method retrieves relevant documents:

```python
from src.shared.python.ai.rag.simple_rag import SimpleRAGStore

rag = SimpleRAGStore()
# ... populate rag with documents ...

session = AssistantSession(provider="ollama", rag_store=rag)
```

Before every response, `AssistantSession` calls `rag_store.query(user_text, 5)`
and prepends the top results to the system prompt.  Failures are caught and
logged; the session continues without RAG context.

---

## Streaming protocol

`send_message` is an async generator compatible with the WebSocket protocol
used by `src/api/routes/chat_ws.py`:

| Property | Value |
|---|---|
| Chunk type | `str` — never `None` |
| Chunk content | Raw text fragment from the provider |
| Ordering | Chronological; first chunk arrives as soon as the provider starts responding |
| Termination | Generator returns (`StopAsyncIteration`) when provider signals completion |
| Tool notices | Interspersed as `\n[Tool <name>: <result>]\n` fragments |
| Error notices | Interspersed as `\n[Error: <message>]` fragments |

### WebSocket integration example

```python
# FastAPI WebSocket handler
@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session = AssistantSession(provider="anthropic")
    while True:
        text = await websocket.receive_text()
        async for chunk in session.send_message(text):
            await websocket.send_text(chunk)
```

---

## Disabling the assistant

Downstream applications can disable the assistant entirely via an environment
variable or configuration flag:

```python
import os
from src.shared.python.ai.assistant_core import AssistantSession

ASSISTANT_ENABLED = os.getenv("ASSISTANT_DISABLED", "0") != "1"

if ASSISTANT_ENABLED:
    session = AssistantSession(provider="ollama")
else:
    session = None  # no-op path; assistant not loaded
```

The headless entrypoint (`src/shared/python/ai/headless_entrypoint.py`) also
respects this pattern and can be used as a reference for CLI integrations.

---

## API reference

### `AssistantSession`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `provider` | `str` | `"ollama"` | Provider name: `"ollama"`, `"openai"`, `"anthropic"`, `"gemini"` |
| `adapter` | `BaseAgentAdapter \| None` | `None` | Pre-built adapter (overrides `provider`) |
| `adapter_kwargs` | `dict \| None` | `None` | Forwarded to the adapter constructor |
| `tool_registry` | `ToolRegistry \| None` | `None` | Tool set available to the AI |
| `rag_store` | `Any \| None` | `None` | Context retrieval store |
| `confirmation_callback` | `async (str, dict) -> bool \| None` | `None` | Gate for confirmation-required tools |
| `system_prompt` | `str \| None` | `None` | Override default system prompt |

#### Methods

- `send_message(text: str) -> AsyncIterator[str]` — stream the response
- `get_history() -> list[Message]` — return a copy of the conversation history
- `reset() -> None` — clear history, retain adapter and registry
