from pathlib import Path

# Patch chat_service.py
chat_service = Path("src/api/services/chat_service.py")
content = chat_service.read_text("utf-8")

# Fix __init__
init_old = """    def __init__(self) -> None:
        self._sessions: OrderedDict[str, ConversationContext] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._adapter: BaseAgentAdapter | None = None
        self._lock = threading.Lock()
        self._load_adapter()"""
init_new = """    def __init__(self) -> None:
        self._sessions: OrderedDict[str, ConversationContext] = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._adapter: BaseAgentAdapter | None = None
        self._lock = threading.Lock()

        from src.shared.python.ai.tool_registry import ToolRegistry
        from src.shared.python.ai.sample_tools import register_golf_suite_tools
        self._tool_registry = ToolRegistry()
        register_golf_suite_tools(self._tool_registry)

        self._load_adapter()"""
content = content.replace(init_old, init_new)

# Fix stream_response signature
sig_old = "    async def stream_response(self, session_id: str) -> AsyncIterator[str]:"
sig_new = "    async def stream_response(self, session_id: str) -> AsyncIterator[Any]:"
content = content.replace(sig_old, sig_new)

# Fix stream logic
stream_old = """        full_response: list[str] = []

        def _run_sync() -> list[str]:
            \"\"\"Run synchronous adapter streaming in thread.\"\"\"
            chunks: list[str] = []
            for chunk in self._adapter.stream_response(  # type: ignore[union-attr]
                temp_ctx.messages[-1].content if temp_ctx.messages else "",
                temp_ctx,
                [],  # No tools for now
            ):
                if chunk.content:
                    chunks.append(chunk.content)
            return chunks

        # Run in thread pool and yield chunks
        # We use a queue-based approach for true streaming
        import queue

        chunk_queue: queue.Queue[str | None] = queue.Queue()

        def _stream_to_queue() -> None:
            try:
                for chunk in self._adapter.stream_response(  # type: ignore[union-attr]
                    "",  # message already in context
                    temp_ctx,
                    [],
                ):
                    if chunk.content:
                        chunk_queue.put(chunk.content)
                        full_response.append(chunk.content)
            except (RuntimeError, ValueError, OSError) as e:
                chunk_queue.put(f"\\n[Error: {e}]")
            finally:
                chunk_queue.put(None)  # Sentinel

        thread = threading.Thread(target=_stream_to_queue, daemon=True)
        thread.start()

        while True:
            try:
                item = await asyncio.to_thread(chunk_queue.get, timeout=60.0)
            except (FileNotFoundError, OSError):
                break
            if item is None:
                break
            yield item

        thread.join(timeout=5.0)

        # Save assistant response to context
        complete_response = "".join(full_response)
        if complete_response:
            with self._lock:
                ctx.add_assistant_message(complete_response)
                self._persist_session(session_id)"""

stream_new = """        import queue
        import json
        from src.shared.python.ai.types import ToolCall

        chunk_queue: queue.Queue[Any] = queue.Queue()

        def _stream_to_queue() -> None:
            try:
                while True:
                    current_response = []
                    tool_calls_accumulator = {}

                    for chunk in self._adapter.stream_response(  # type: ignore[union-attr]
                        "",  # message already in context
                        temp_ctx,
                        self._tool_registry.declarations(),
                    ):
                        if chunk.content:
                            chunk_queue.put(chunk.content)
                            current_response.append(chunk.content)

                        if chunk.tool_call_delta:
                            for tc in chunk.tool_call_delta.get("tool_calls", []):
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_accumulator:
                                    tool_calls_accumulator[idx] = {
                                        "id": tc.get("id"),
                                        "name": tc.get("function", {}).get("name", ""),
                                        "arguments": tc.get("function", {}).get("arguments", "")
                                    }
                                else:
                                    if tc.get("function", {}).get("arguments"):
                                        tool_calls_accumulator[idx]["arguments"] += tc["function"]["arguments"]

                    complete_response = "".join(current_response)
                    
                    tool_calls = []
                    for idx, tc_data in tool_calls_accumulator.items():
                        try:
                            args = json.loads(tc_data["arguments"] or "{}")
                        except json.JSONDecodeError:
                            args = {"raw": tc_data["arguments"]}
                        tool_calls.append(ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args))

                    if complete_response or tool_calls:
                        with self._lock:
                            ctx.add_assistant_message(complete_response, tool_calls=tool_calls)
                            temp_ctx.add_assistant_message(complete_response, tool_calls=tool_calls)
                            self._persist_session(session_id)

                    if not tool_calls:
                        break

                    for tc in tool_calls:
                        chunk_queue.put({"type": "tool_call_started", "tool": tc.name})
                        
                        try:
                            tool_result = self._tool_registry.execute(tc.name, tc.arguments)
                            result_str = str(tool_result.result) if tool_result.success else str(tool_result.error)
                            success = tool_result.success
                        except Exception as e:
                            result_str = str(e)
                            success = False
                            
                        chunk_queue.put({
                            "type": "tool_call_result" if success else "tool_error",
                            "tool": tc.name,
                            "detail": result_str
                        })

                        with self._lock:
                            ctx.add_tool_result(tc.id, result_str)
                            temp_ctx.add_tool_result(tc.id, result_str)
                            self._persist_session(session_id)
                            
            except Exception as e:
                chunk_queue.put(f"\\n[Error: {e}]")
            finally:
                chunk_queue.put(None)  # Sentinel

        thread = threading.Thread(target=_stream_to_queue, daemon=True)
        thread.start()

        while True:
            try:
                item = await asyncio.to_thread(chunk_queue.get, timeout=60.0)
            except (FileNotFoundError, OSError):
                break
            if item is None:
                break
            yield item

        thread.join(timeout=5.0)"""

content = content.replace(stream_old, stream_new)
chat_service.write_text(content, "utf-8")
print("Patched chat_service.py")

# Patch chat_ws.py
chat_ws = Path("src/api/routes/chat_ws.py")
content = chat_ws.read_text("utf-8")

ws_old = """                # Stream response chunks
                async for chunk in chat_service.stream_response(session_id):
                    await websocket.send_json({"type": "chunk", "content": chunk})"""
ws_new = """                # Stream response chunks
                async for chunk in chat_service.stream_response(session_id):
                    if isinstance(chunk, dict):
                        await websocket.send_json(chunk)
                    else:
                        await websocket.send_json({"type": "chunk", "content": str(chunk)})"""
content = content.replace(ws_old, ws_new)
chat_ws.write_text(content, "utf-8")
print("Patched chat_ws.py")
