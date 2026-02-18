# ADR-0003: WebSocket Protocol for Real-Time Simulation

- Status: Accepted
- Date: 2026-02-18
- Decision Makers: Core maintainers
- Related Issues/PRs: #1486

## Context

Physics simulations produce continuous state data (joint angles, forces, torques,
contact points) that needs to be streamed to the UI in real-time for
visualization and interactive control. HTTP request/response polling would
introduce unacceptable latency for 60+ Hz simulation updates. The protocol must
also support bidirectional communication so users can adjust simulation
parameters while it is running.

Key constraints:

- Simulation state updates at 60-1000 Hz depending on engine and model complexity
- UI must render smoothly without dropped frames or lag spikes
- Users need interactive control (pause, resume, parameter adjustment) during simulation
- Multiple simultaneous simulations may run for cross-engine comparison

## Decision

Use WebSocket connections for real-time simulation data streaming:

- Each simulation session gets its own WebSocket connection
- Binary messages (MessagePack) for high-frequency state updates to minimize serialization overhead
- JSON messages for control commands and configuration changes
- Server-side rate limiting to prevent client overwhelm when simulation runs faster than render rate
- Structured message types with a `type` field for routing

## Alternatives Considered

1. **HTTP long polling**: Simple to implement but adds per-request overhead.
   Not suitable for high-frequency updates.
2. **Server-Sent Events (SSE)**: Server-to-client only; cannot send control
   commands back without a separate HTTP channel.
3. **gRPC streaming**: Excellent performance but adds protobuf compilation step
   and is harder to debug from browser-based tools.

## Consequences

- **Positive**: Low-latency simulation visualization (sub-millisecond message delivery)
- **Positive**: Bidirectional communication enables interactive parameter tuning
- **Positive**: Per-session connections provide natural isolation for multi-engine comparison
- **Negative**: Connection management complexity (reconnection, error handling, cleanup)
- **Negative**: Requires careful handling of connection lifecycle and resource cleanup
- **Follow-ups**: Define message schema versioning strategy for protocol evolution

## Validation

- WebSocket integration tests in `tests/api/` verify message round-trip latency
- Load tests simulate multiple concurrent WebSocket connections
- Connection lifecycle tests verify proper cleanup on disconnect and server shutdown
