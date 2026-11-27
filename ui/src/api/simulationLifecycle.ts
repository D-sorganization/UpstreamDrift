/**
 * Simulation WebSocket lifecycle reducer (#14950).
 *
 * Owns run identity and phase; socket handlers dispatch tagged events and stale
 * runIds are ignored.
 */

export type SimulationPhase =
  | "idle"
  | "connecting"
  | "starting"
  | "running"
  | "paused"
  | "stopping"
  | "completed"
  | "stopped"
  | "failed"
  | "faulted"
  | "disconnected"
  | "lost";

export interface SimulationLifecycleState {
  runId: number;
  phase: SimulationPhase;
  paused: boolean;
  failureMessage: string | null;
}

export type SimulationRunLifecycle = SimulationLifecycleState;

export const initialSimulationLifecycle: SimulationLifecycleState = {
  runId: 0,
  phase: "idle",
  paused: false,
  failureMessage: null,
};

export type SimulationLifecycleAction =
  | { type: "begin_connect"; runId: number }
  | { type: "socket_open"; runId: number }
  | { type: "server_running"; runId: number }
  | { type: "server_paused"; runId: number }
  | { type: "server_resumed"; runId: number }
  | { type: "server_complete"; runId: number }
  | { type: "server_stopped"; runId: number }
  | { type: "server_error"; runId: number; message: string }
  | { type: "server_fault"; runId: number; message: string }
  | { type: "socket_lost"; runId: number; message: string }
  | { type: "socket_disconnected"; runId: number }
  | { type: "socket_clean_close"; runId: number }
  | { type: "user_stopping"; runId: number }
  | { type: "user_reset"; runId: number };

function isCurrentRun(
  state: SimulationLifecycleState,
  runId: number,
): boolean {
  return runId === state.runId;
}

export function simulationLifecycleReducer(
  state: SimulationLifecycleState,
  action: SimulationLifecycleAction,
): SimulationLifecycleState {
  if (action.type === "begin_connect") {
    if (action.runId <= state.runId) {
      return state;
    }
    return {
      runId: action.runId,
      phase: "connecting",
      paused: false,
      failureMessage: null,
    };
  }

  if (!isCurrentRun(state, action.runId)) {
    return state;
  }

  switch (action.type) {
    case "socket_open":
      return { ...state, phase: "starting", paused: false, failureMessage: null };
    case "server_running":
      return { ...state, phase: "running", paused: false };
    case "server_paused":
      if (state.phase !== "running" && state.phase !== "paused") {
        return state;
      }
      return { ...state, phase: "paused", paused: true };
    case "server_resumed":
      if (state.phase !== "running" && state.phase !== "paused") {
        return state;
      }
      return { ...state, phase: "running", paused: false };
    case "user_stopping":
      return { ...state, phase: "stopping" };
    case "server_complete":
      return { ...state, phase: "completed", paused: false };
    case "server_stopped":
      return { ...state, phase: "stopped", paused: false };
    case "server_error":
      return {
        ...state,
        phase: "failed",
        paused: false,
        failureMessage: action.message,
      };
    case "server_fault":
      return {
        ...state,
        phase: "faulted",
        paused: false,
        failureMessage: action.message,
      };
    case "socket_lost":
      return {
        ...state,
        phase: "lost",
        paused: false,
        failureMessage: action.message,
      };
    case "socket_disconnected":
      return {
        ...state,
        phase: "disconnected",
        paused: false,
      };
    case "socket_clean_close":
      if (
        state.phase === "failed" ||
        state.phase === "lost" ||
        state.phase === "faulted"
      ) {
        return state;
      }
      return {
        ...state,
        phase:
          state.phase === "running" || state.phase === "paused"
            ? "stopped"
            : "idle",
        paused: false,
      };
    case "user_reset":
      return { ...state, phase: "idle", paused: false, failureMessage: null };
    default:
      return state;
  }
}

export const reduceSimLifecycle = simulationLifecycleReducer;

export function deriveIsRunning(phase: SimulationPhase): boolean {
  return phase === "running" || phase === "paused";
}

export function deriveIsPaused(
  phase: SimulationPhase,
  paused: boolean,
): boolean {
  return phase === "paused" || (phase === "running" && paused);
}
