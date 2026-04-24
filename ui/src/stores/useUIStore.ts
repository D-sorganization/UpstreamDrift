/**
 * UI Store — Global UI State Management
 *
 * Centralizes transient UI state like panel visibility,
 * sidebar collapse state, and theme preferences.
 *
 * @module stores/useUIStore
 */

import { create } from 'zustand';

// ── Types ─────────────────────────────────────────────────────────────────

export type ExpertiseLevel = 'beginner' | 'intermediate' | 'advanced' | 'expert';

export interface UIStoreState {
  /** Whether the help panel is visible */
  helpOpen: boolean;
  /** Topic/term ID to scroll the help panel to when opened */
  helpTopicId: string | null;
  /** Whether the diagnostics panel is visible */
  diagnosticsOpen: boolean;
  /** Left sidebar collapsed */
  leftSidebarCollapsed: boolean;
  /** Right sidebar collapsed */
  rightSidebarCollapsed: boolean;
  /** User expertise level (drives glossary language and chat context). */
  expertiseLevel: ExpertiseLevel;
}

export interface UIStoreActions {
  /** Toggle help panel */
  toggleHelp: () => void;
  /** Set help panel state explicitly */
  setHelpOpen: (open: boolean) => void;
  /** Open the help panel scrolled to a specific topic/term. */
  openHelpPanel: (topicId: string) => void;
  /** Toggle diagnostics panel */
  toggleDiagnostics: () => void;
  /** Set diagnostics panel state explicitly */
  setDiagnosticsOpen: (open: boolean) => void;
  /** Toggle left sidebar */
  toggleLeftSidebar: () => void;
  /** Toggle right sidebar */
  toggleRightSidebar: () => void;
  /** Set expertise level. */
  setExpertiseLevel: (level: ExpertiseLevel) => void;
  /** Reset UI to defaults */
  resetUI: () => void;
}

export type UIStore = UIStoreState & UIStoreActions;

// ── Store ─────────────────────────────────────────────────────────────────

export const useUIStore = create<UIStore>((set) => ({
  helpOpen: false,
  helpTopicId: null,
  diagnosticsOpen: false,
  leftSidebarCollapsed: false,
  rightSidebarCollapsed: false,
  expertiseLevel: 'intermediate',

  toggleHelp: () => set((state) => ({ helpOpen: !state.helpOpen })),
  setHelpOpen: (open) => set({ helpOpen: open }),
  openHelpPanel: (topicId) => set({ helpOpen: true, helpTopicId: topicId }),

  toggleDiagnostics: () =>
    set((state) => ({ diagnosticsOpen: !state.diagnosticsOpen })),
  setDiagnosticsOpen: (open) => set({ diagnosticsOpen: open }),

  toggleLeftSidebar: () =>
    set((state) => ({ leftSidebarCollapsed: !state.leftSidebarCollapsed })),
  toggleRightSidebar: () =>
    set((state) => ({ rightSidebarCollapsed: !state.rightSidebarCollapsed })),

  setExpertiseLevel: (level) => set({ expertiseLevel: level }),

  resetUI: () =>
    set({
      helpOpen: false,
      helpTopicId: null,
      diagnosticsOpen: false,
      leftSidebarCollapsed: false,
      rightSidebarCollapsed: false,
      expertiseLevel: 'intermediate',
    }),
}));
