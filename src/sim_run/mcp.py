"""Model Context Protocol (MCP) stdio adapter for headless simulation runs.

Epic #14085 / Child Task #14092; FastMCP stdio transport (#15022).
Thin MCP adapter exposing simulation run service tools:
describe_engines, submit_run, get_run, fetch_artifact; plus the putting-study
tools optimize_putt, putting_misread_cost and putting_library_study (#14177,
see ``putting_tools``) and the putting-core tools putting_simulate_putt,
putting_read_green, putting_stimpmeter and putting_capture_speed (#14141 step
9a, see ``putting_core_tools``).
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent_api import (
    SERVER_NAME,
    engines_for_describe,
    read_auth_token,
    read_private_mode,
)
from .agent_capability_tools import (
    agent_capability_tool_handlers,
    agent_capability_tool_specs,
)
from .fleet_agent_api import CONTRACT_VERSION, client_run_record_v1
from .models import RunRequest, export_schemas
from .putting_core_tools import core_tool_handlers, core_tool_specs
from .putting_tools import putting_tool_handlers, putting_tool_specs
from .store import RunNotFoundError, SimulationRunStore
from .worker import SimulationBusyError, SimulationWorkerPool

_LOG = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_VERSION = "1.0.0"


def _tool_spec(
    name: str, desc: str, props: dict[str, str], req: list[str]
) -> dict[str, Any]:
    p = {k: {"type": v} for k, v in props.items()}
    return {
        "name": name,
        "description": desc,
        "inputSchema": {"type": "object", "properties": p, "required": req},
    }


@dataclass
class SecurityPolicy:
    """Service boundary security policy enforcing Private Mode and auth checks."""

    private_mode: bool = False
    auth_token: str | None = None
    allowed_roots: list[Path] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> SecurityPolicy:
        return cls(private_mode=read_private_mode(), auth_token=read_auth_token())

    def check_auth(self, token: str | None) -> None:
        if self.auth_token is not None and token != self.auth_token:
            raise PermissionError("Unauthorized: invalid or missing auth token")

    def check_execution_allowed(self) -> None:
        if self.private_mode:
            raise PermissionError("Private Mode active: simulation execution denied")


class SimulationMcpServer:
    """Simulation MCP tool router (JSON-RPC test harness + FastMCP backend)."""

    def __init__(
        self,
        store: SimulationRunStore | None = None,
        worker_pool: SimulationWorkerPool | None = None,
        security_policy: SecurityPolicy | None = None,
        results_root: Path | str | None = None,
    ) -> None:
        if store is not None:
            self.store = store
        else:
            self.store = SimulationRunStore(
                results_root=Path(results_root or "./sim_results")
            )
        self.worker_pool = worker_pool or SimulationWorkerPool(self.store)
        self.security = security_policy or SecurityPolicy.from_env()

    def tool_describe_engines(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.security.check_auth(arguments.get("auth_token"))
        request_schema = export_schemas()
        payload = {
            "contract_version": CONTRACT_VERSION,
            "engines": engines_for_describe(),
            "request_schema": request_schema,
            "schema_version": "1.0.0",
        }
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}

    def tool_submit_run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.security.check_auth(arguments.get("auth_token"))
        self.security.check_execution_allowed()
        req_payload = arguments.get("request") or arguments
        cleaned = {k: v for k, v in req_payload.items() if k != "auth_token"}
        try:
            request = RunRequest.model_validate(cleaned)
        except Exception as exc:
            raise ValueError(f"Invalid RunRequest payload: {exc}") from exc

        record = self.store.create_run(request)
        try:
            finished = self.worker_pool.execute_run(record)
        except SimulationBusyError as exc:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"SimulationBusyError: {exc}"}],
            }

        payload = client_run_record_v1(finished.model_dump(mode="json"))
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}

    def tool_get_run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.security.check_auth(arguments.get("auth_token"))
        run_id = arguments.get("run_id")
        if not run_id or not isinstance(run_id, str):
            raise ValueError("Parameter 'run_id' is required and must be a string")
        try:
            record = self.store.get_run(run_id)
        except RunNotFoundError as exc:
            raise ValueError(f"Run '{run_id}' not found") from exc
        payload = client_run_record_v1(record.model_dump(mode="json"))
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}

    def tool_fetch_artifact(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.security.check_auth(arguments.get("auth_token"))
        run_id, name = arguments.get("run_id"), arguments.get("artifact_name")
        if (
            not run_id
            or not name
            or not isinstance(run_id, str)
            or not isinstance(name, str)
        ):
            raise ValueError(
                "Parameters 'run_id' and 'artifact_name' are required strings"
            )
        if ".." in name.replace("\\", "/").split("/") or Path(name).is_absolute():
            raise ValueError("Rejecting path traversal attempt in artifact_name")

        run_dir = self.store.get_run_dir(run_id)
        target = (run_dir / "artifacts" / name).resolve()
        if not str(target).startswith(str((run_dir / "artifacts").resolve())):
            raise ValueError("Artifact path escapes run artifacts directory")
        if not target.is_file():
            raise FileNotFoundError(f"Artifact '{name}' not found for run '{run_id}'")

        try:
            content, enc = target.read_text(encoding="utf-8"), "utf-8"
        except UnicodeDecodeError:
            content, enc = (
                base64.b64encode(target.read_bytes()).decode("ascii"),
                "base64",
            )
        payload = {
            "run_id": run_id,
            "artifact_name": name,
            "size_bytes": target.stat().st_size,
            "encoding": enc,
            "content": content,
        }
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}

    def _tool_handlers(self) -> dict[str, Any]:
        return {
            "describe_engines": self.tool_describe_engines,
            "submit_run": self.tool_submit_run,
            "get_run": self.tool_get_run,
            "fetch_artifact": self.tool_fetch_artifact,
            **putting_tool_handlers(self.security.check_auth),
            **core_tool_handlers(self.security.check_auth),
            **agent_capability_tool_handlers(self.security.check_auth, self.store),
        }

    def tool_specs(self) -> list[dict[str, Any]]:
        """Pinned MCP tool list shared by JSON-RPC tests and FastMCP ``tools/list``."""
        a = {"auth_token": "string"}
        run_p = {
            **a,
            "engine_type": "string",
            "duration": "number",
            "timestep": "number",
        }
        return [
            _tool_spec("describe_engines", "List engines and schema", a, []),
            _tool_spec("submit_run", "Submit simulation run", run_p, ["engine_type"]),
            _tool_spec(
                "get_run", "Retrieve run by ID", {**a, "run_id": "string"}, ["run_id"]
            ),
            _tool_spec(
                "fetch_artifact",
                "Fetch simulation artifact",
                {**a, "run_id": "string", "artifact_name": "string"},
                ["run_id", "artifact_name"],
            ),
            *putting_tool_specs(),
            *core_tool_specs(),
            *agent_capability_tool_specs(),
        ]

    def _tools(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for :meth:`tool_specs`."""
        return self.tool_specs()

    def invoke_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a single tool call; raises on auth and validation errors."""
        handlers = self._tool_handlers()
        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: '{name}'")
        return handler(arguments or {})

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a single JSON-RPC 2.0 message."""
        method, params, req_id = (
            request.get("method"),
            request.get("params") or {},
            request.get("id"),
        )
        try:
            if method == "initialize":
                res: Any = {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                }
            elif method in ("notifications/initialized", "ping"):
                res = {}
            elif method == "tools/list":
                res = {"tools": self.tool_specs()}
            elif method == "tools/call":
                res = self.invoke_tool(
                    str(params.get("name")), params.get("arguments") or {}
                )
            else:
                raise ValueError(f"Unknown JSON-RPC method: '{method}'")
            return {"jsonrpc": "2.0", "id": req_id, "result": res}
        except PermissionError as exc:
            _LOG.warning("mcp security refusal: %s", exc)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }
        except (ValueError, FileNotFoundError) as exc:
            _LOG.warning("mcp validation error: %s", exc)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": str(exc)},
            }
        except Exception as exc:  # noqa: BLE001 — JSON-RPC error envelope
            _LOG.exception("mcp internal error: %s", exc)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }


async def run_stdio_loop(server: SimulationMcpServer) -> None:
    """Run the official MCP SDK stdio loop via FastMCP."""
    from .mcp_fastmcp import build_fastmcp_server

    fastmcp = build_fastmcp_server(server)
    if fastmcp is None:
        raise RuntimeError(
            "ud-sim-mcp: install the MCP SDK: pip install 'upstream-drift[sim-mcp]'"
        )
    await fastmcp.run_stdio_async()


def main() -> None:  # pragma: no cover — CLI process entrypoint
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    from .mcp_fastmcp import build_fastmcp_server

    fastmcp = build_fastmcp_server(SimulationMcpServer())
    if fastmcp is None:
        sys.stderr.write(
            "ud-sim-mcp: install the MCP SDK: pip install 'upstream-drift[sim-mcp]'\n"
        )
        raise SystemExit(2)
    fastmcp.run(transport="stdio")
    raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
