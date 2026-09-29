import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { BACKEND_PORT, getApiBase, isTauri } from './backend';

/**
 * Tests for the backend helper module (issue #6637).
 * Verifies port consistency and API base URL resolution for Tauri vs. browser mode.
 */

describe('backend.ts – issue #6637', () => {
  const EXPECTED_PORT = 8000;

  describe('BACKEND_PORT constant', () => {
    it('equals the canonical Python default port (8000)', () => {
      expect(BACKEND_PORT).toBe(EXPECTED_PORT);
    });
  });

  describe('getApiBase()', () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    afterEach(() => {
      // Restore window to original state
      if ('__TAURI_INTERNALS__' in window) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (window as any).__TAURI_INTERNALS__;
      }
    });

    it('returns empty string in browser (non-Tauri) mode', () => {
      // Ensure Tauri is not present
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (window as any).__TAURI_INTERNALS__;
      expect(isTauri()).toBe(false);
      expect(getApiBase()).toBe('');
    });

    it('returns http://localhost:BACKEND_PORT in Tauri mode', () => {
      // Simulate Tauri environment
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__TAURI_INTERNALS__ = {};
      expect(isTauri()).toBe(true);
      expect(getApiBase()).toBe(`http://localhost:${BACKEND_PORT}`);
    });

    it('Tauri API base includes the correct canonical port', () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__TAURI_INTERNALS__ = {};
      const base = getApiBase();
      expect(base).toContain(String(EXPECTED_PORT));
    });
  });
});
