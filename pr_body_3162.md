Resolves #3162 from the G-track adversarial review.

## Changes

### Core: `chat_service.py`

- Injected `ToolRegistry` into `ChatService.__init__`, pre-populated with `register_golf_suite_tools()`
- Replaced hard-coded empty tool list with dynamic `ToolDeclaration` list built from registry
- Implemented agentic execution loop in `stream_response`: streams adapter chunks, accumulates tool call deltas, executes tools with `concurrent.futures` timeout, appends results back to context, and loops until no more tool calls
- Changed return type annotation from `AsyncIterator[str]` to `AsyncIterator[Any]` to support structured event dicts alongside text chunks

### Protocol: `chat_ws.py`

- WebSocket handler now dispatches structured dict events (`tool_call_started`, `tool_call_result`, `tool_error`) directly, while still wrapping plain text chunks

### Config: `ai/config.py`

- Added `TOOL_TIMEOUT` env var and `get_tool_timeout()` accessor (default 30s) for safe tool execution gating

### Bug Fix: `contracts/decorators.py`

- Resolved merge conflict that left conflict markers in the file, breaking syntax and adding async precondition support

## Tests

- `tests/test_chat_service_tools.py`: mocked adapter yields a tool call; asserts two-pass streaming, `tool_call_started` and `tool_call_result` events, and correct text on second pass
- `tests/test_chat_service_tool_timeout.py`: registers a slow tool; patches `get_tool_timeout()` to 0.1s; asserts `tool_error` with timeout message in detail field

Both tests pass cleanly.
