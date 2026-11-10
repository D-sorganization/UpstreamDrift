import { describe, expect, it } from "vitest";
import {
  deriveIsPaused,
  deriveIsRunning,
  initialSimulationLifecycle,
  simulationLifecycleReducer,
} from "./simulationLifecycle";

describe("simulationLifecycleReducer", () => {
  it("ignores events from a stale runId", () => {
    let state = initialSimulationLifecycle;
    state = simulationLifecycleReducer(state, { type: "begin_connect", runId: 1 });
    state = simulationLifecycleReducer(state, { type: "socket_open", runId: 1 });
    state = simulationLifecycleReducer(state, { type: "server_running", runId: 2 });
    expect(state.phase).toBe("starting");
    expect(deriveIsRunning(state.phase)).toBe(false);
  });

  it("tracks pause inside running", () => {
    let state = initialSimulationLifecycle;
    state = simulationLifecycleReducer(state, { type: "begin_connect", runId: 1 });
    state = simulationLifecycleReducer(state, { type: "socket_open", runId: 1 });
    state = simulationLifecycleReducer(state, { type: "server_running", runId: 1 });
    state = simulationLifecycleReducer(state, { type: "server_paused", runId: 1 });
    expect(deriveIsPaused(state.phase, state.paused)).toBe(true);
    state = simulationLifecycleReducer(state, { type: "server_stopped", runId: 1 });
    expect(deriveIsPaused(state.phase, state.paused)).toBe(false);
  });
});
