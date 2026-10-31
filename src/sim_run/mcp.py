"""Model Context Protocol (MCP) stdio adapter for headless simulation runs.

Epic #14085 / Child Task #14092:
Thin stdio MCP adapter exposing 4 simulation run service tools:
describe_engines, submit_run, get_run, fetch_artifact. Epic #14141 steps
9a/9b add 5 putting computation tools (see :mod:`.putting_tools`).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import putting_tools
from .models import RunRequest, export_schemas
from .store import RunNotFoundError, SimulationRunStore
from .worker import SimulationBusyError, SimulationWorkerPool

_LOG = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ud-sim-mcp"
SERVER_VERSION = "1.0.0"

_ENGINES = [
    ("pendulum", "Double pendulum dynamics benchmark", ["rigid_body", "kinematics"]),
    ("mujoco", "MuJoCo physics engine", ["rigid_body", "contacts"]),
    ("pinocchio", "Pinocchio rigid body algorithms", ["rigid_body", "dynamics"]),
    ("drake", "Drake multibody plant simulation", ["rigid_body", "contacts"]),
    ("opensim", "OpenSim musculoskeletal biomechanics", ["muscles", "biomechanics"]),
]
_ENGINE_METADATA = [
    {"engine_type": e, "description": d, "capabilities": c} for e, d, c in _ENGINES
]


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
        p_mode = os.environ.get("UD_PRIVATE_MODE", "").lower() in ("1", "true", "yes")
        return cls(
            private_mode=p_mode, auth_token=os.environ.get("UD_SIM_AUTH_TOKEN") or None
        )

    def check_auth(self, token: str | None) -> None:
        if self.auth_token is not None and token != self.auth_token:
            raise PermissionError("Unauthorized: invalid or missing auth token")

    def check_execution_allowed(self) -> None:
        if self.private_mode:
            raise PermissionError("Private Mode active: simulation execution denied")


class SimulationMcpServer:
    """Stdio JSON-RPC 2.0 MCP server exposing simulation run tools."""

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
        payload = {
            "engines": _ENGINE_METADATA,
            "request_schema": export_schemas(),
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

        return {
            "content": [
                {"type": "text", "text": json.dumps(finished.model_dump(mode="json"))}
            ]
        }

    def tool_get_run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.security.check_auth(arguments.get("auth_token"))
        run_id = arguments.get("run_id")
        if not run_id or not isinstance(run_id, str):
            raise ValueError("Parameter 'run_id' is required and must be a string")
        try:
            record = self.store.get_run(run_id)
        except RunNotFoundError as exc:
            raise ValueError(f"Run '{run_id}' not found") from exc
        return {
            "content": [
                {"type": "text", "text": json.dumps(record.model_dump(mode="json"))}
            ]
        }

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

    def _tools(self) -> list[dict[str, Any]]:
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
            *putting_tools.tool_specs(),
        ]

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
                res = {"tools": self._tools()}
            elif method == "tools/call":
                # Putting tools are pure computations (no filesystem/process
                # side effects), so only auth applies; Private Mode's
                # check_execution_allowed() never gates them.
                handlers = {
                    "describe_engines": self.tool_describe_engines,
                    "submit_run": self.tool_submit_run,
                    "get_run": self.tool_get_run,
                    "fetch_artifact": self.tool_fetch_artifact,
                    "putting_simulate_putt": lambda args: (
                        putting_tools.tool_putting_simulate_putt(self.security, args)
                    ),
                    "putting_read_green": lambda args: (
                        putting_tools.tool_putting_read_green(self.security, args)
                    ),
                    "putting_stimpmeter": lambda args: (
                        putting_tools.tool_putting_stimpmeter(self.security, args)
                    ),
                    "putting_capture_speed": lambda args: (
                        putting_tools.tool_putting_capture_speed(self.security, args)
                    ),
                    "putting_make_probability": lambda args: (
                        putting_tools.tool_putting_make_probability(self.security, args)
                    ),
                }
                handler = handlers.get(str(params.get("name")))
                if not handler:
                    raise ValueError(f"Unknown tool: '{params.get('name')}'")
                res = handler(params.get("arguments") or {})
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
    """Run asynchronous stdin/stdout JSON-RPC event loop."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )
    w_transport, w_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(w_transport, w_protocol, None, loop)
    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            req = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue
        resp = server.handle_request(req)
        writer.write((json.dumps(resp) + "\n").encode("utf-8"))
        await writer.drain()


def main() -> None:  # pragma: no cover — CLI process entrypoint
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    asyncio.run(run_stdio_loop(SimulationMcpServer()))


if __name__ == "__main__":  # pragma: no cover
    main()
