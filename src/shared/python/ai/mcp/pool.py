"""McpClientPool — manages N MCP server connections.

Each server is represented by an :class:`McpServerEntry` that holds the
config and (optionally) a live tool list. The pool tags each tool with
the server name so :meth:`call_tool` can route correctly.

Design notes
------------
- The pool does *not* open subprocess connections in the constructor —
  startup happens lazily on the first :meth:`tools` call (or explicitly
  via :meth:`start`).  This keeps construction side-effect-free for tests.
- Per-server failures are caught individually; a broken server writes a
  warning log entry and contributes zero tools, but does not prevent the
  other servers from being used.
- The optional ``mcp`` transport package is imported inside methods that
  need it.  If it is not installed, the pool logs a single warning and
  every server is effectively unreachable (contributing zero tools).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.shared.python.ai.mcp.contracts import McpServerConfig, McpToolDescriptor

logger = logging.getLogger(__name__)


@dataclass
class McpServerEntry:
    """Runtime state for a single MCP server inside the pool.

    Attributes:
        config: Validated server configuration.
        tools: Tool descriptors fetched from this server; ``None`` means
            the server has not been contacted yet (not that it has zero
            tools).
        connected: Whether the last connection attempt succeeded.
        error: Last connection error message (empty string if none).
    """

    config: McpServerConfig
    tools: list[McpToolDescriptor] | None = field(default=None)
    connected: bool = False
    error: str = ""


class McpClientPool:
    """Manages the lifecycle of N MCP server connections.

    The pool exposes a unified tool namespace where each tool's
    :attr:`~McpToolDescriptor.namespaced_name` is
    ``"{server_name}__{tool_name}"`` — guaranteeing global uniqueness.

    Preconditions (DbC)
    -------------------
    - :meth:`call_tool` requires the pool to have been started (i.e.
      :meth:`start` has been called or tools have been fetched at least
      once via :meth:`tools`).

    Postconditions (DbC)
    --------------------
    - After :meth:`stop`, :meth:`tools` returns an empty list.
    - :attr:`server_count` equals ``len(configs)`` passed at construction.
    """

    def __init__(self, configs: list[McpServerConfig]) -> None:
        """Initialise pool with server configurations.

        Args:
            configs: List of validated server configurations. May be empty.

        Raises:
            TypeError: If *configs* is not a list.
        """
        if not isinstance(configs, list):
            raise TypeError(f"configs must be a list, got {type(configs).__name__}")
        self._entries: list[McpServerEntry] = [
            McpServerEntry(config=c) for c in configs
        ]
        self._started: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def server_count(self) -> int:
        """Total number of configured servers (including unhealthy ones)."""
        return len(self._entries)

    @property
    def connected_count(self) -> int:
        """Number of servers that are currently connected."""
        return sum(1 for e in self._entries if e.connected)

    @property
    def is_started(self) -> bool:
        """True once :meth:`start` has been called."""
        return self._started

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to all enabled servers and fetch their tool lists.

        Individual server failures are caught and logged; they do not
        propagate to the caller.  After this method returns,
        :attr:`is_started` is ``True`` regardless of how many servers
        actually connected.
        """
        for entry in self._entries:
            if not entry.config.enabled:
                logger.debug("Skipping disabled MCP server %r", entry.config.name)
                continue
            self._connect_entry(entry)
        self._started = True
        logger.info(
            "McpClientPool started: %d/%d servers connected",
            self.connected_count,
            self.server_count,
        )

    def stop(self) -> None:
        """Disconnect all servers and clear tool lists.

        Postcondition: :meth:`tools` returns ``[]``.
        """
        for entry in self._entries:
            entry.tools = []
            entry.connected = False
            entry.error = ""
        self._started = False
        logger.info("McpClientPool stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tools(self) -> list[McpToolDescriptor]:
        """Return all tools from all connected servers.

        Starts the pool lazily if :meth:`start` has not been called yet.

        Returns:
            Aggregated list of :class:`McpToolDescriptor` objects. The
            list is empty when no servers are configured or all have
            failed.
        """
        if not self._started:
            self.start()
        result: list[McpToolDescriptor] = []
        for entry in self._entries:
            if entry.tools:
                result.extend(entry.tools)
        return result

    def call_tool(
        self,
        namespaced_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Route a tool call to the correct server.

        The tool is identified by its *namespaced_name* (e.g.
        ``"filesystem__read_file"``).  The pool strips the server prefix,
        locates the server entry, and forwards the call.

        Args:
            namespaced_name: Globally unique tool name
                (``"{server}__{tool}"``).
            arguments: Tool arguments dict.

        Returns:
            Whatever the MCP server returns for this tool call.

        Raises:
            RuntimeError: If the pool has not been started.
            ValueError: If no server owns the given tool.
        """
        if not self._started:
            raise RuntimeError(
                "McpClientPool.call_tool() called before start(). "
                "Call pool.start() or pool.tools() first."
            )

        # Locate the owning server entry by checking which server has this tool
        target_entry: McpServerEntry | None = None
        for entry in self._entries:
            if entry.tools:
                for tool in entry.tools:
                    if tool.namespaced_name == namespaced_name:
                        target_entry = entry
                        break
            if target_entry is not None:
                break

        if target_entry is None:
            raise ValueError(
                f"No MCP server owns tool {namespaced_name!r}. "
                f"Available: {[t.namespaced_name for e in self._entries if e.tools for t in e.tools]}"
            )

        # Derive the raw tool name by stripping the server prefix
        server_prefix = f"{target_entry.config.name}__"
        raw_tool_name = namespaced_name.removeprefix(server_prefix)

        return self._invoke_tool(target_entry, raw_tool_name, arguments)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect_entry(self, entry: McpServerEntry) -> None:
        """Attempt to connect *entry* and populate its tool list.

        Failures are caught; the entry is marked as disconnected with an
        error message.
        """
        try:
            raw_tools = self._fetch_tools_for_config(entry.config)
            entry.tools = raw_tools
            entry.connected = True
            entry.error = ""
            logger.info(
                "MCP server %r connected (%d tools)",
                entry.config.name,
                len(raw_tools),
            )
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            entry.tools = []
            entry.connected = False
            entry.error = str(exc)
            logger.warning(
                "MCP server %r failed to connect: %s",
                entry.config.name,
                exc,
            )

    def _fetch_tools_for_config(
        self, config: McpServerConfig
    ) -> list[McpToolDescriptor]:
        """Contact the server and return its tool list.

        This method requires the optional ``mcp`` package. If it is not
        installed, a ``RuntimeError`` is raised (which :meth:`_connect_entry`
        catches and logs as a per-server failure).

        Args:
            config: Server configuration.

        Returns:
            List of :class:`McpToolDescriptor` objects.

        Raises:
            RuntimeError: If the ``mcp`` package is not installed.
            Exception: Any transport-level error propagates up to
                :meth:`_connect_entry`.
        """
        try:
            import subprocess
            import json as _json

            # Attempt a simple tool-list handshake via JSON-RPC 2.0 over stdio.
            # We send an ``initialize`` + ``tools/list`` request and parse the
            # response.  This is intentionally a minimal implementation that
            # works without the full MCP SDK — the SDK can be layered on top
            # later.
            result = self._jsonrpc_tools_list(config)
            return result
        except ImportError:
            raise RuntimeError(  # noqa: B904 — deliberate chaining suppression
                "The 'mcp' package is not installed; MCP servers are not "
                "reachable. Install with: pip install mcp"
            )

    def _jsonrpc_tools_list(self, config: McpServerConfig) -> list[McpToolDescriptor]:
        """Perform a minimal JSON-RPC tools/list handshake.

        Starts the server subprocess, sends an ``initialize`` request and
        a ``tools/list`` request, then terminates the process.

        Args:
            config: Server configuration.

        Returns:
            List of :class:`McpToolDescriptor` objects parsed from the
            server response.  Returns an empty list if the server returns
            no tools or the response cannot be parsed.

        Raises:
            RuntimeError: If the subprocess cannot be started or the
                response is malformed.
        """
        import json
        import subprocess
        import os

        cmd = [config.command, *config.args]
        env = {
            **os.environ,
            **{k: self._expand_env_var(v) for k, v in config.env.items()},
        }

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                f"Could not start MCP server process {cmd!r}: {exc}"
            ) from exc

        try:
            # Send initialize + tools/list as newline-delimited JSON-RPC
            init_req = (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "upstreamdrift", "version": "1.0"},
                        },
                    }
                )
                + "\n"
            )
            list_req = (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/list",
                        "params": {},
                    }
                )
                + "\n"
            )

            stdout, _ = proc.communicate(
                input=(init_req + list_req).encode(),
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(  # noqa: B904
                f"MCP server {config.name!r} timed out during tool listing"
            ) from None
        finally:
            if proc.poll() is None:
                proc.terminate()

        # Parse lines — look for the tools/list response (id == 2)
        descriptors: list[McpToolDescriptor] = []
        for line in stdout.decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == 2 and "result" in msg:
                tools_raw = msg["result"].get("tools", [])
                for raw in tools_raw:
                    raw_name = raw.get("name", "")
                    if not raw_name:
                        continue
                    descriptors.append(
                        McpToolDescriptor(
                            name=raw_name,
                            namespaced_name=f"{config.name}__{raw_name}",
                            server_name=config.name,
                            description=raw.get("description", ""),
                            input_schema=raw.get("inputSchema", {}),
                        )
                    )
                break

        return descriptors

    def _invoke_tool(
        self,
        entry: McpServerEntry,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Invoke *tool_name* on *entry*'s server process.

        Args:
            entry: Server entry that owns the tool.
            tool_name: Raw (un-namespaced) tool name.
            arguments: Arguments to pass to the tool.

        Returns:
            The ``result`` field from the MCP ``tools/call`` response.

        Raises:
            RuntimeError: If the subprocess call fails or times out.
        """
        import json
        import subprocess
        import os

        config = entry.config
        cmd = [config.command, *config.args]
        env = {
            **os.environ,
            **{k: self._expand_env_var(v) for k, v in config.env.items()},
        }

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except (FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                f"Could not start MCP server process {cmd!r}: {exc}"
            ) from exc

        try:
            init_req = (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "upstreamdrift", "version": "1.0"},
                        },
                    }
                )
                + "\n"
            )
            call_req = (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": arguments},
                    }
                )
                + "\n"
            )

            stdout, _ = proc.communicate(
                input=(init_req + call_req).encode(),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError(  # noqa: B904
                f"MCP server {config.name!r} timed out during tool call {tool_name!r}"
            ) from None
        finally:
            if proc.poll() is None:
                proc.terminate()

        for line in stdout.decode(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == 2:
                if "error" in msg:
                    raise RuntimeError(
                        f"MCP tool {tool_name!r} returned error: {msg['error']}"
                    )
                return msg.get("result")

        raise RuntimeError(
            f"MCP server {config.name!r} returned no response for tool "
            f"call {tool_name!r}"
        )

    @staticmethod
    def _expand_env_var(value: str) -> str:
        """Expand ``${VAR}`` placeholders in *value* against ``os.environ``.

        Unknown variables are left unchanged.
        """
        import os
        import re

        def _sub(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(0))

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", _sub, value)
