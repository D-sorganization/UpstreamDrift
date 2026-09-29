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

export interface UIStoreState {
  /** Whether the help panel is visible */
  helpOpen: boolean;
  /** Whether the diagnostics panel is visible */
  diagnosticsOpen: boolean;
  /** Left sidebar collapsed */
  leftSidebarCollapsed: boolean;
  /** Right sidebar collapsed */
  rightSidebarCollapsed: boolean;
}

export interface UIStoreActions {
  /** Toggle help panel */
  toggleHelp: () => void;
  /** Set help panel state explicitly */
  setHelpOpen: (open: boolean) => void;
  /** Toggle diagnostics panel */
  toggleDiagnostics: () => void;
  /** Set diagnostics panel state explicitly */
  setDiagnosticsOpen: (open: boolean) => void;
  /** Toggle left sidebar */
  toggleLeftSidebar: () => void;
  /** Toggle right sidebar */
  toggleRightSidebar: () => void;
  /** Reset UI to defaults */
  resetUI: () => void;
}

export type UIStore = UIStoreState & UIStoreActions;

// ── Store ─────────────────────────────────────────────────────────────────

export const useUIStore = create<UIStore>((set) => ({
  helpOpen: false,
  diagnosticsOpen: false,
  leftSidebarCollapsed: false,
  rightSidebarCollapsed: false,

  toggleHelp: () => set((state) => ({ helpOpen: !state.helpOpen })),
  setHelpOpen: (open) => set({ helpOpen: open }),

  toggleDiagnostics: () =>
    set((state) => ({ diagnosticsOpen: !state.diagnosticsOpen })),
  setDiagnosticsOpen: (open) => set({ diagnosticsOpen: open }),

  toggleLeftSidebar: () =>
    set((state) => ({ leftSidebarCollapsed: !state.leftSidebarCollapsed })),
  toggleRightSidebar: () =>
    set((state) => ({ rightSidebarCollapsed: !state.rightSidebarCollapsed })),

  resetUI: () =>
    set({
      helpOpen: false,
      diagnosticsOpen: false,
      leftSidebarCollapsed: false,
      rightSidebarCollapsed: false,
    }),
}));
