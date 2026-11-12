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
5. `run_counterfactual_ztcf_golden` / `run_golf_simulator_golden_shot` (Refs #15021): pinned golden
   cases routed through :mod:`src.sim_run.agent_capabilities`, with v1 capability run records,
   qualification banners, and 64 KiB inline payload limits (larger results spill to artifacts).
   REST mirrors the same handlers under ``/tools/agent-sim-run/``.

## Host Configuration

Cursor and Codex discover the repo-local [`.mcp.json`](../../.mcp.json) entry
`ud-sim-mcp`, which launches the `ud-sim-mcp` console script after
`pip install -e .`. Other MCP hosts can use the same command:

```json
{
  "mcpServers": {
    "ud-sim-mcp": {
      "command": "ud-sim-mcp",
      "args": [],
      "env": {
        "UPSTREAMDRIFT_PRIVATE_MODE": "0",
        "UPSTREAMDRIFT_AGENT_API_TOKEN": "optional-token"
      }
    }
  }
}
```

Legacy `UD_PRIVATE_MODE` and `UD_SIM_AUTH_TOKEN` remain accepted aliases.

Regenerate the committed capability inventory with
`python3 -m scripts.generate_sim_run_mcp_inventory` (see
[`sim_run_mcp_capability_inventory.md`](sim_run_mcp_capability_inventory.md)).

## Security Model & Boundary Gates

- **Private Mode**: When `UPSTREAMDRIFT_PRIVATE_MODE=1` (or legacy `UD_PRIVATE_MODE=1`) is active, calling `submit_run` raises `PermissionError("Private Mode active: simulation execution denied")` (code `-32000`).
- **Traversal Guard**: Artifact requests with `..` or absolute paths are rejected with `ValueError`.
