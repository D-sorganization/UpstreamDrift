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
  /** Whether parameters were touched in-session (user edit or hydration) */
  parametersTouched: boolean;
  /** Whether server-side defaults were already applied this session */
  defaultsHydrated: boolean;
  /** Server-side trajectory recording state (issue #7452) */
  recording: RecordingState;
}

export interface SimulationStoreActions {
  /** Update simulation parameters (partial merge) */
  setParameters: (params: Partial<SimulationParameters>) => void;
  /** Replace all simulation parameters */
  replaceParameters: (params: SimulationParameters) => void;
  /**
   * Apply server-side simulation defaults (settings, #7457) at app start.
   *
   * Only takes effect once per session and never clobbers an in-session
   * change: if the user already edited parameters (or a hydration already
   * ran), this is a no-op (#7424 parameter-reset bug guard).
   */
  hydrateDefaults: (defaults: Partial<Pick<SimulationParameters, 'duration' | 'timestep'>>) => void;
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
  parametersTouched: false,
  defaultsHydrated: false,
  recording: { ...DEFAULT_RECORDING },

  setParameters: (partial) =>
    set((state) => ({
      parameters: { ...state.parameters, ...partial },
      parametersTouched: true,
    })),

  replaceParameters: (params) =>
    set({ parameters: params, parametersTouched: true }),

  hydrateDefaults: (defaults) =>
    set((state) => {
      if (state.defaultsHydrated || state.parametersTouched || state.hasRun) {
        // Never clobber an in-session change (#7424) or re-hydrate.
        return { defaultsHydrated: true };
      }
      return {
        parameters: { ...state.parameters, ...defaults },
        defaultsHydrated: true,
      };
    }),

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
      parametersTouched: false,
      recording: { ...DEFAULT_RECORDING },
    }),
}));
