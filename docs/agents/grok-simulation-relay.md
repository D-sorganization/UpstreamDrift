# Local Grok Bot Simulation Relay Plan

Status: proposal for UpstreamDrift #14337. This document defines the bounded
caller path and its falsifying acceptance test. It does not authorize dispatch,
deployment, a new transport, or changes to the simulation service.

## Proposed Caller Path

Use the existing same-host stdio MCP adapter from
[`docs/development/sim_run_mcp.md`](../development/sim_run_mcp.md): the Grok Bot
MCP client launches `python -m src.sim_run.mcp` with the UpstreamDrift checkout
as its working directory and exchanges MCP JSON-RPC over the child process's
standard input and output. The adapter calls the existing run service. It must
not start a second runner or route through a network endpoint.

This path is conditional on the current Grok Bot host supporting a local stdio
MCP server configuration. Confirm that capability before implementation. If it
cannot launch the local command, stop and request the specific owner decision;
do not substitute a remote hub, HTTP relay, or listener.

The only simulation tools in scope are:

| Tool | Allowed Purpose |
| --- | --- |
| `describe_engines` | Read the registered engine capabilities and request schema. |
| `submit_run` | Submit a schema-valid request through the existing service and isolated worker. |
| `get_run` | Read a run record by its opaque run ID. |
| `fetch_artifact` | Read an artifact by run ID and artifact name, within that run's artifact directory. |

Do not add `cancel_run`, `list_runs`, arbitrary filesystem reads, another
execution lifecycle, or engine-specific shortcuts in this relay.

## Trust And Security Boundary

- Keep ingress on the same host and on stdio. Do not bind a TCP, UDP, or Unix
  socket for Grok Bot and do not bridge to the remote hub.
- Keep authentication and execution policy in the existing service boundary.
  When `UD_SIM_AUTH_TOKEN` is configured, the client must supply the valid token
  to each tool call. Do not put the token in a prompt, command-line argument, or
  log; use the host's approved environment-secret handling.
- Do not override `UD_PRIVATE_MODE` in the client configuration. With Private
  Mode active, `submit_run` must fail at the service boundary. Read-only tools
  remain subject to authentication and the existing service policy.
- Accept only the existing request schema. Reject missing or invalid run IDs,
  unknown tools, missing/invalid authentication, and artifact names that are
  absolute or traverse outside the run's artifact directory.
- Preserve the service's existing worker isolation, limits, durable run IDs,
  result validation, and provenance. The relay must not claim correctness from
  provenance alone.

## Falsifying Acceptance Test

Exercise the proposed stdio MCP path with a deterministic test-only engine and
temporary run store; do not use private user data or a live engine. The test
must fail if any of these assertions fails:

1. A client initializes the adapter, lists the four allowed tools, submits one
   valid fixture run, and receives a run ID with a successful status and result.
2. `get_run` returns the same terminal status and result for that ID, and the
   run's declared artifact reference can be fetched by its run ID and artifact
   name.
3. Missing or incorrect authentication is rejected when a token is configured;
   a `submit_run` call is rejected when Private Mode is active.
4. `fetch_artifact` rejects an absolute path and a `..` traversal attempt, and
   an unknown operation such as `cancel_run` is not exposed or dispatched.
5. The stdio round-trip binds no network socket. At the host-integration stage,
   the configured Grok Bot MCP client must launch the same local command and
   pass the same fixture scenario; if the host cannot do this, the path is not
   accepted and the owner decision in the preceding section is required.

The existing adapter tests already cover the in-process four-tool round-trip,
Private Mode rejection, authentication, traversal rejection, and no socket
bind. Those tests are useful lower-level evidence, but they do not by
themselves prove that the Grok Bot host can launch the configured stdio child.
Record that host check separately; do not report it as passed until it runs.

## Scope Gate

Implementation may begin only after the existing Grok Bot host-launch
capability is confirmed and the ordinary issue claim and acceptance gates are
current. Any request for remote transport, a listener, new authorization policy,
additional tools, or deployment requires a separate explicit owner decision.
