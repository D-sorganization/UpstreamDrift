/**
 * Stores — Barrel re-export for all Zustand stores.
 *
 * Usage:
 *   import { useEngineStore, useSimulationStore, useUIStore } from '@/stores';
 */

export {
  useEngineStore,
  selectLoadedEngines,
  selectEffectiveEngine,
  selectEngine,
} from './useEngineStore';
export type {
  EngineLoadState,
  ManagedEngine,
  EngineStore,
  EngineStoreState,
  EngineStoreActions,
} from './useEngineStore';

export {
  useSimulationStore,
  DEFAULT_PARAMETERS,
} from './useSimulationStore';
export type {
  SimulationParameters,
  SimulationStore,
  SimulationStoreState,
  SimulationStoreActions,
} from './useSimulationStore';

export {
  useUIStore,
} from './useUIStore';
export type {
  UIStore,
  UIStoreState,
  UIStoreActions,
} from './useUIStore';

export {
  useChatStore,
} from './useChatStore';
export type {
  ChatMessage,
  MessageRole,
  ChatStore,
  ChatStoreState,
  ChatStoreActions,
} from './useChatStore';
