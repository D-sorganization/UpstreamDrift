/**
 * Glossary Store — Caches glossary definitions fetched from the backend.
 *
 * Backed by the `/glossary/{term_id}` endpoint served by the (parallel-agent)
 * glossary route. Results are cached in-memory keyed by `{term_id}@{level}`
 * so that repeated hovers don't re-fetch.
 *
 * See issue #3165.
 */

import { create } from 'zustand';

// ── Types ─────────────────────────────────────────────────────────────────

export type ExpertiseLevel = 'beginner' | 'intermediate' | 'advanced' | 'expert';

export interface Definition {
  term_id: string;
  title: string;
  short: string;
  description?: string;
  level: ExpertiseLevel;
}

export interface GlossaryStoreState {
  cache: Record<string, Definition>;
}

export interface GlossaryStoreActions {
  /** Fetch a definition, caching the result. */
  fetch: (termId: string, level?: ExpertiseLevel) => Promise<Definition | null>;
  /** Synchronous cache lookup. */
  get: (termId: string, level?: ExpertiseLevel) => Definition | undefined;
  /** Clear the cache (primarily for tests). */
  clear: () => void;
}

export type GlossaryStore = GlossaryStoreState & GlossaryStoreActions;

function cacheKey(termId: string, level: ExpertiseLevel): string {
  return `${termId}@${level}`;
}

// ── Store ─────────────────────────────────────────────────────────────────

export const useGlossaryStore = create<GlossaryStore>((set, get) => ({
  cache: {},

  get: (termId, level = 'intermediate') => {
    if (!termId) return undefined;
    return get().cache[cacheKey(termId, level)];
  },

  fetch: async (termId, level = 'intermediate') => {
    if (!termId) return null;
    const key = cacheKey(termId, level);
    const cached = get().cache[key];
    if (cached) return cached;

    try {
      const response = await fetch(
        `/glossary/${encodeURIComponent(termId)}?level=${encodeURIComponent(level)}`,
      );
      if (!response.ok) {
        return null;
      }
      const data = (await response.json()) as Definition;
      set((state) => ({ cache: { ...state.cache, [key]: data } }));
      return data;
    } catch {
      return null;
    }
  },

  clear: () => set({ cache: {} }),
}));
