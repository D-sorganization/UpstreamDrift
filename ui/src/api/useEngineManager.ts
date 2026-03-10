/**
 * @deprecated Use `useEngineStore` from `@/stores/useEngineStore` instead.
 *
 * This module is a backward-compatibility shim. The canonical engine state
 * management is the Zustand store at `@/stores/useEngineStore`.
 *
 * Migration:
 *   Old: import { ManagedEngine, useEngineManager } from '@/api/useEngineManager';
 *   New: import { useEngineStore, type ManagedEngine } from '@/stores/useEngineStore';
 */

// Re-export types from the canonical location
export type { ManagedEngine, EngineLoadState } from '@/stores/useEngineStore';

// Re-export the ENGINE_REGISTRY from the store (keep backward compat)
// The store's registry is internal, so we keep a copy here for direct users.
import { useEngineStore } from '@/stores/useEngineStore';

/**
 * @deprecated Use `useEngineStore()` directly instead.
 *
 * This hook wraps the Zustand store for backward compatibility.
 * It will be removed in a future version.
 */
export function useEngineManager() {
  const engines = useEngineStore((s) => s.engines);
  const requestLoad = useEngineStore((s) => s.requestLoad);
  const unloadEngine = useEngineStore((s) => s.unloadEngine);

  return {
    engines,
    loadedEngines: engines.filter((e) => e.loadState === 'loaded'),
    getEngine: (name: string) => engines.find((e) => e.name === name),
    requestLoad,
    unloadEngine,
  };
}
