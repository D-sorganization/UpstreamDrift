/**
 * Onboarding dismissal persistence adapter (issue #7459).
 *
 * Currently backed by localStorage (per-browser). When server-side user
 * settings land (#7457), swap the implementation of this module for an
 * API-backed one — the OnboardingOverlay only depends on this interface,
 * so nothing else changes.
 */

const STORAGE_KEY = 'upstreamdrift.onboarding.dismissed';

export interface OnboardingPersistence {
  isDismissed(): boolean;
  dismiss(): void;
  /** Test/support helper: clear the dismissal so onboarding shows again. */
  reset(): void;
}

interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

// In-memory fallback for environments without a working localStorage
// (privacy mode, sandboxed iframes, partial test shims).
const memoryStore = new Map<string, string>();
const memoryFallback: KeyValueStore = {
  getItem: (key) => memoryStore.get(key) ?? null,
  setItem: (key, value) => {
    memoryStore.set(key, value);
  },
};

function resolveStore(): KeyValueStore {
  try {
    const ls = window.localStorage;
    if (
      ls &&
      typeof ls.getItem === 'function' &&
      typeof ls.setItem === 'function'
    ) {
      return ls;
    }
  } catch {
    // Storage unavailable — fall through to memory.
  }
  return memoryFallback;
}

export const onboardingPersistence: OnboardingPersistence = {
  isDismissed(): boolean {
    return resolveStore().getItem(STORAGE_KEY) === 'true';
  },
  dismiss(): void {
    resolveStore().setItem(STORAGE_KEY, 'true');
  },
  reset(): void {
    // setItem('false') instead of removeItem: keeps the required store
    // surface minimal (get/set only).
    resolveStore().setItem(STORAGE_KEY, 'false');
  },
};
