---
issue: 14950
summary: "Replace simulation boolean soup with lifecycle reducer, stale-socket identity guard, and ordered speed control"
---

## Handoff

- Completes #14950 (Closes #14950, Part of #14934):
  - Added pure reducer transitions and discriminated lifecycle phase union (`SimulationPhase` / `SimulationRunLifecycle` / `reduceSimLifecycle`).
  - Added stale-socket identity guard (`ws !== wsRef.current`) across all socket callbacks.
  - Guaranteed double-start guard while connecting.
  - Ordered speed changes with pause/stop actions over the WebSocket stream.
  - Backwards-compatible derived getters: `isRunning`, `isPaused`, `isConnected`, `error`, `status.phase`.
  - Added unit test suite in `ui/src/hooks/useSimulation.test.ts`.
