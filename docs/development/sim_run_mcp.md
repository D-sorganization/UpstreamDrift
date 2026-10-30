# Simulation Run MCP Adapter (Headless Engine Service)

Child task of Epic [#14085](https://github.com/D-sorganization/UpstreamDrift/issues/14085) (Refs [#14092](https://github.com/D-sorganization/UpstreamDrift/issues/14092)).

## Overview

The `src.sim_run.mcp` module provides a thin **stdio Model Context Protocol (MCP)** server for local agents (Grok Bot, Claude Desktop, Antigravity) to query physics simulation engines, execute runs, poll results, and inspect artifacts.

- **No network sockets**: Communicates strictly over stdio; zero listeners bound (ADR-0007).
- **Service boundary enforcement**: Auth tokens and Private Mode restrictions enforced at service boundary.
- **Durable isolated execution**: Isolated worker subprocesses via `SimulationWorkerPool` and durable records via `SimulationRunStore`.

## Tool Surface

1. `describe_engines`: Lists registered engines, capabilities, and JSON Schema definitions for `RunRequest`.
2. `submit_run`: Validates schema, executes solver in isolated subprocess, validates numerical integrity, and returns record.
3. `get_run`: Retrieves durable record, status, timestamps, and result envelope by `run_id`.
4. `fetch_artifact`: Safely reads simulation artifacts with strict path traversal defenses.

## Host Configuration

Add to client configuration (e.g. `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "upstreamdrift-sim": {
      "command": "python",
      "args": ["-m", "src.sim_run.mcp"],
      "env": {
        "PYTHONPATH": ".",
        "UD_PRIVATE_MODE": "0",
        "UD_SIM_AUTH_TOKEN": "optional-token"
      }
    }
  }
}
```

## Security Model & Boundary Gates

- **Private Mode**: When `UD_PRIVATE_MODE=1` is active, calling `submit_run` raises `PermissionError("Private Mode active: simulation execution denied")` (code `-32000`).
- **Traversal Guard**: Artifact requests with `..` or absolute paths are rejected with `ValueError`.
