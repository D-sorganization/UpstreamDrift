/**
 * Simulation Store — Global Simulation State Management
 *
 * Centralizes simulation parameters and run configuration.
 * This store owns the "what to simulate" state while the actual
 * WebSocket connection / frame streaming stays in the useSimulation hook.
 *
 * @module stores/useSimulationStore
 */

import { create } from 'zustand';

// ── Types ─────────────────────────────────────────────────────────────────

export interface SimulationParameters {
  duration: number;
  timestep: number;
  liveAnalysis: boolean;
  gpuAcceleration: boolean;
}

/** Trajectory recording lifecycle (issue #7452). */
export type RecordingStatus = 'idle' | 'recording' | 'saved';

export interface RecordingState {
  status: RecordingStatus;
  /** Frames captured in the last completed recording. */
  frameCount: number;
}

export interface SimulationStoreState {
  /** Current simulation parameters */
  parameters: SimulationParameters;
  /** Whether a simulation has been started at least once */
  hasRun: boolean;
  /** Server-side trajectory recording state (issue #7452) */
  recording: RecordingState;
}

export interface SimulationStoreActions {
  /** Update simulation parameters (partial merge) */
  setParameters: (params: Partial<SimulationParameters>) => void;
  /** Replace all simulation parameters */
  replaceParameters: (params: SimulationParameters) => void;
  /** Mark that a simulation has been started */
  markRun: () => void;
  /** Mark server-side recording as active (issue #7452) */
  startRecording: () => void;
  /** Mark recording stopped, remembering how many frames were saved */
  finishRecording: (frameCount: number) => void;
  /** Clear recording state back to idle */
  resetRecording: () => void;
  /** Reset to defaults */
  resetParameters: () => void;
}

export type SimulationStore = SimulationStoreState & SimulationStoreActions;

// ── Defaults ──────────────────────────────────────────────────────────────

export const DEFAULT_PARAMETERS: SimulationParameters = {
  duration: 3.0,
  timestep: 0.002,
  liveAnalysis: true,
  gpuAcceleration: false,
};

export const DEFAULT_RECORDING: RecordingState = {
  status: 'idle',
  frameCount: 0,
};

// ── Store ─────────────────────────────────────────────────────────────────

export const useSimulationStore = create<SimulationStore>((set) => ({
  parameters: { ...DEFAULT_PARAMETERS },
  hasRun: false,
  recording: { ...DEFAULT_RECORDING },

  setParameters: (partial) =>
    set((state) => ({
      parameters: { ...state.parameters, ...partial },
    })),

  replaceParameters: (params) => set({ parameters: params }),

  markRun: () => set({ hasRun: true }),

  startRecording: () =>
    set({ recording: { status: 'recording', frameCount: 0 } }),

  finishRecording: (frameCount) =>
    set({ recording: { status: 'saved', frameCount } }),

  resetRecording: () => set({ recording: { ...DEFAULT_RECORDING } }),

  resetParameters: () =>
    set({
      parameters: { ...DEFAULT_PARAMETERS },
      hasRun: false,
      recording: { ...DEFAULT_RECORDING },
    }),
}));
