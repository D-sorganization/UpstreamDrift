import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import { useUIStore } from './useUIStore';

describe('useUIStore', () => {
  beforeEach(() => {
    act(() => {
      useUIStore.getState().resetUI();
    });
  });

  describe('initial state', () => {
    it('starts with help panel closed', () => {
      expect(useUIStore.getState().helpOpen).toBe(false);
    });

    it('starts with diagnostics panel closed', () => {
      expect(useUIStore.getState().diagnosticsOpen).toBe(false);
    });

    it('starts with sidebars expanded', () => {
      expect(useUIStore.getState().leftSidebarCollapsed).toBe(false);
      expect(useUIStore.getState().rightSidebarCollapsed).toBe(false);
    });
  });

  describe('help panel', () => {
    it('toggleHelp opens when closed', () => {
      act(() => {
        useUIStore.getState().toggleHelp();
      });
      expect(useUIStore.getState().helpOpen).toBe(true);
    });

    it('toggleHelp closes when open', () => {
      act(() => {
        useUIStore.getState().setHelpOpen(true);
      });
      act(() => {
        useUIStore.getState().toggleHelp();
      });
      expect(useUIStore.getState().helpOpen).toBe(false);
    });

    it('setHelpOpen sets explicit state', () => {
      act(() => {
        useUIStore.getState().setHelpOpen(true);
      });
      expect(useUIStore.getState().helpOpen).toBe(true);

      act(() => {
        useUIStore.getState().setHelpOpen(false);
      });
      expect(useUIStore.getState().helpOpen).toBe(false);
    });
  });

  describe('diagnostics panel', () => {
    it('toggleDiagnostics opens when closed', () => {
      act(() => {
        useUIStore.getState().toggleDiagnostics();
      });
      expect(useUIStore.getState().diagnosticsOpen).toBe(true);
    });

    it('setDiagnosticsOpen sets explicit state', () => {
      act(() => {
        useUIStore.getState().setDiagnosticsOpen(true);
      });
      expect(useUIStore.getState().diagnosticsOpen).toBe(true);
    });
  });

  describe('sidebars', () => {
    it('toggleLeftSidebar collapses when expanded', () => {
      act(() => {
        useUIStore.getState().toggleLeftSidebar();
      });
      expect(useUIStore.getState().leftSidebarCollapsed).toBe(true);
    });

    it('toggleRightSidebar collapses when expanded', () => {
      act(() => {
        useUIStore.getState().toggleRightSidebar();
      });
      expect(useUIStore.getState().rightSidebarCollapsed).toBe(true);
    });

    it('toggleLeftSidebar expands when collapsed', () => {
      act(() => {
        useUIStore.getState().toggleLeftSidebar();
      });
      act(() => {
        useUIStore.getState().toggleLeftSidebar();
      });
      expect(useUIStore.getState().leftSidebarCollapsed).toBe(false);
    });
  });

  describe('resetUI', () => {
    it('resets all UI state to defaults', () => {
      act(() => {
        useUIStore.getState().setHelpOpen(true);
        useUIStore.getState().setDiagnosticsOpen(true);
        useUIStore.getState().toggleLeftSidebar();
        useUIStore.getState().toggleRightSidebar();
      });

      act(() => {
        useUIStore.getState().resetUI();
      });

      const state = useUIStore.getState();
      expect(state.helpOpen).toBe(false);
      expect(state.diagnosticsOpen).toBe(false);
      expect(state.leftSidebarCollapsed).toBe(false);
      expect(state.rightSidebarCollapsed).toBe(false);
    });
  });
});
