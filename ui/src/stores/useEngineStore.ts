/**
 * Engine Store — Global Engine State Management
 *
 * Centralizes physics engine selection, loading, and status.
 * Replaces per-component `useEngineManager()` calls with a single
 * source of truth that can be consumed from any component.
 *
 * @module stores/useEngineStore
 */

import { create } from 'zustand';

// ── Types ─────────────────────────────────────────────────────────────────

export type EngineLoadState = 'idle' | 'loading' | 'loaded' | 'error';

export interface ManagedEngine {
  name: string;
  displayName: string;
  description: string;
  loadState: EngineLoadState;
  available: boolean;
  version?: string;
  capabilities: string[];
  error?: string;
}

export interface EngineStoreState {
  /** All known engines with their current load state */
  engines: ManagedEngine[];
  /** Currently selected engine name (null = none) */
  selectedEngine: string | null;
}

export interface EngineStoreActions {
  /** Select an engine by name */
  selectEngine: (name: string | null) => void;
  /** Request loading an engine (async) */
  requestLoad: (name: string) => Promise<void>;
  /** Unload an engine */
  unloadEngine: (name: string) => void;
  /** Reset all engines to idle */
  resetEngines: () => void;
}

export type EngineStore = EngineStoreState & EngineStoreActions;

// ── Static Registry ───────────────────────────────────────────────────────

const ENGINE_REGISTRY: Omit<ManagedEngine, 'loadState' | 'available' | 'error'>[] = [
  {
    name: 'mujoco',
    displayName: 'MuJoCo',
    description: 'High-performance physics for robotics and biomechanics',
    capabilities: ['rigid_body', 'contact', 'tendons', 'actuators'],
  },
  {
    name: 'drake',
    displayName: 'Drake',
    description: 'Optimization-based multibody dynamics',
    capabilities: ['rigid_body', 'optimization', 'contact'],
  },
  {
    name: 'pinocchio',
    displayName: 'Pinocchio',
    description: 'Rigid body dynamics algorithms',
    capabilities: ['rigid_body', 'inverse_kinematics'],
  },
  {
    name: 'opensim',
    displayName: 'OpenSim',
    description: 'Musculoskeletal modeling and biomechanics simulation',
    capabilities: ['musculoskeletal', 'inverse_kinematics', 'muscle_analysis'],
  },
  {
    name: 'myosuite',
    displayName: 'MyoSuite',
    description: 'Muscle-tendon control and neural activation',
    capabilities: ['musculoskeletal', 'muscle_control', 'neural_activation'],
  },
  {
    name: 'putting_green',
    displayName: 'Putting Green',
    description: 'Golf putting green with ball roll physics',
    capabilities: ['surface_modeling', 'ball_physics', 'terrain'],
  },
];

// ── API helpers ───────────────────────────────────────────────────────────

interface EngineStatus {
  available: boolean;
  version?: string;
  capabilities?: string[];
}

async function probeEngine(engineName: string): Promise<EngineStatus> {
  const response = await fetch(`/api/engines/${engineName}/probe`);
  if (!response.ok) {
    throw new Error(`Failed to probe engine: ${engineName}`);
  }
  return response.json();
}

async function loadEngineApi(engineName: string): Promise<EngineStatus> {
  const response = await fetch(`/api/engines/${engineName}/load`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error(`Failed to load engine: ${engineName}`);
  }
  return response.json();
}

// ── Initial state ─────────────────────────────────────────────────────────

function createInitialEngines(): ManagedEngine[] {
  return ENGINE_REGISTRY.map((e) => ({
    ...e,
    loadState: 'idle' as EngineLoadState,
    available: true,
  }));
}

// ── Store ─────────────────────────────────────────────────────────────────

export const useEngineStore = create<EngineStore>((set, get) => ({
  engines: createInitialEngines(),
  selectedEngine: null,

  selectEngine: (name) => set({ selectedEngine: name }),

  requestLoad: async (engineName) => {
    // Set to loading
    set((state) => ({
      engines: state.engines.map((e) =>
        e.name === engineName
          ? { ...e, loadState: 'loading' as EngineLoadState, error: undefined }
          : e
      ),
    }));

    try {
      const probeResult = await probeEngine(engineName);

      if (!probeResult.available) {
        set((state) => ({
          engines: state.engines.map((e) =>
            e.name === engineName
              ? {
                  ...e,
                  loadState: 'error' as EngineLoadState,
                  available: false,
                  error: 'Engine not installed on this system',
                }
              : e
          ),
        }));
        return;
      }

      const loadResult = await loadEngineApi(engineName);

      set((state) => ({
        engines: state.engines.map((e) =>
          e.name === engineName
            ? {
                ...e,
                loadState: 'loaded' as EngineLoadState,
                available: true,
                version: loadResult.version,
                capabilities: loadResult.capabilities || e.capabilities,
              }
            : e
        ),
      }));

      // Auto-select if nothing is selected
      if (!get().selectedEngine) {
        set({ selectedEngine: engineName });
      }
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Unknown error loading engine';
      set((state) => ({
        engines: state.engines.map((e) =>
          e.name === engineName
            ? { ...e, loadState: 'error' as EngineLoadState, error: message }
            : e
        ),
      }));
    }
  },

  unloadEngine: (engineName) => {
    const { selectedEngine } = get();
    set((state) => ({
      engines: state.engines.map((e) =>
        e.name === engineName
          ? { ...e, loadState: 'idle' as EngineLoadState, error: undefined }
          : e
      ),
      selectedEngine:
        selectedEngine === engineName ? null : selectedEngine,
    }));
  },

  resetEngines: () =>
    set({
      engines: createInitialEngines(),
      selectedEngine: null,
    }),
}));

// ── Derived selectors ─────────────────────────────────────────────────────

/** Select only loaded engines (avoids re-renders when unrelated state changes) */
export const selectLoadedEngines = (state: EngineStore): ManagedEngine[] =>
  state.engines.filter((e) => e.loadState === 'loaded');

/** Select the effectively active engine (selected or first loaded) */
export const selectEffectiveEngine = (state: EngineStore): string | null => {
  const { selectedEngine, engines } = state;
  if (selectedEngine) {
    const eng = engines.find(
      (e) => e.name === selectedEngine && e.loadState === 'loaded'
    );
    if (eng) return selectedEngine;
  }
  const firstLoaded = engines.find((e) => e.loadState === 'loaded');
  return firstLoaded ? firstLoaded.name : null;
};

/** Get a specific engine by name */
export const selectEngine = (name: string) => (state: EngineStore): ManagedEngine | undefined =>
  state.engines.find((e) => e.name === name);
