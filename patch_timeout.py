from pathlib import Path

# Patch chat_service.py
chat_service = Path("src/api/services/chat_service.py")
content = chat_service.read_text("utf-8")

old_exec = """                    for tc in tool_calls:
                        chunk_queue.put({"type": "tool_call_started", "tool": tc.name})
                        
                        try:
                            tool_result = self._tool_registry.execute(tc.name, tc.arguments)
                            result_str = str(tool_result.result) if tool_result.success else str(tool_result.error)
                            success = tool_result.success
                        except Exception as e:
                            result_str = str(e)
                            success = False"""

new_exec = """                    import concurrent.futures
                    from src.shared.python.ai.config import get_tool_timeout
                    for tc in tool_calls:
                        chunk_queue.put({"type": "tool_call_started", "tool": tc.name})
                        
                        try:
                            timeout_sec = get_tool_timeout()
                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                future = executor.submit(self._tool_registry.execute, tc.name, tc.arguments)
                                tool_result = future.result(timeout=timeout_sec)
                            result_str = str(tool_result.result) if tool_result.success else str(tool_result.error)
                            success = tool_result.success
                        except concurrent.futures.TimeoutError:
                            result_str = f"Tool execution timed out after {timeout_sec}s"
                            success = False
                        except Exception as e:
                            result_str = str(e)
                            success = False"""

content = content.replace(old_exec, new_exec)
chat_service.write_text(content, "utf-8")

# Also add get_tool_timeout to src/shared/python/ai/config.py
config = Path("src/shared/python/ai/config.py")
config_content = config.read_text("utf-8")
if "get_tool_timeout" not in config_content:
    config_content += """\n
ENV_TOOL_TIMEOUT = "TOOL_TIMEOUT"
DEFAULT_TOOL_TIMEOUT = 30.0

def get_tool_timeout() -> float:
    \"\"\"Get maximum tool execution timeout.

    Returns:
        Timeout in seconds from TOOL_TIMEOUT or default.
    \"\"\"
    return (
        get_env_float(ENV_TOOL_TIMEOUT, default=DEFAULT_TOOL_TIMEOUT)
        or DEFAULT_TOOL_TIMEOUT
    )
"""
    config.write_text(config_content, "utf-8")

print("Done")
