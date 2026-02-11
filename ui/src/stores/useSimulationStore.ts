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

export interface SimulationStoreState {
  /** Current simulation parameters */
  parameters: SimulationParameters;
  /** Whether a simulation has been started at least once */
  hasRun: boolean;
}

export interface SimulationStoreActions {
  /** Update simulation parameters (partial merge) */
  setParameters: (params: Partial<SimulationParameters>) => void;
  /** Replace all simulation parameters */
  replaceParameters: (params: SimulationParameters) => void;
  /** Mark that a simulation has been started */
  markRun: () => void;
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

// ── Store ─────────────────────────────────────────────────────────────────

export const useSimulationStore = create<SimulationStore>((set) => ({
  parameters: { ...DEFAULT_PARAMETERS },
  hasRun: false,

  setParameters: (partial) =>
    set((state) => ({
      parameters: { ...state.parameters, ...partial },
    })),

  replaceParameters: (params) => set({ parameters: params }),

  markRun: () => set({ hasRun: true }),

  resetParameters: () =>
    set({
      parameters: { ...DEFAULT_PARAMETERS },
      hasRun: false,
    }),
}));
