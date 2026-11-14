import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  BACKEND_PORT,
  DEFAULT_BACKEND_URL,
  getApiBase,
  isTauri,
  resolveBackendBaseUrl,
  checkHealth,
  type BackendStatus,
} from './backend';

/**
 * Tests for the backend helper module (issues #6637, #14951).
 * Verifies port consistency, dynamic backend port resolution, and health probing.
 */

describe('backend.ts – issue #6637 & #14951', () => {
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

  describe('dynamic backend port resolution (#14951)', () => {
    it('defaults DEFAULT_BACKEND_URL to 127.0.0.1:8000', () => {
      expect(DEFAULT_BACKEND_URL).toBe('http://127.0.0.1:8000');
    });

    it('resolveBackendBaseUrl returns DEFAULT_BACKEND_URL when status is undefined or null', () => {
      expect(resolveBackendBaseUrl()).toBe('http://127.0.0.1:8000');
      expect(resolveBackendBaseUrl(null)).toBe('http://127.0.0.1:8000');
    });

    it('resolveBackendBaseUrl returns DEFAULT_BACKEND_URL when port is null, 0, or negative', () => {
      expect(resolveBackendBaseUrl({ port: null })).toBe('http://127.0.0.1:8000');
      expect(resolveBackendBaseUrl({ port: 0 })).toBe('http://127.0.0.1:8000');
      expect(resolveBackendBaseUrl({ port: -1 })).toBe('http://127.0.0.1:8000');
    });

    it('resolveBackendBaseUrl returns dynamic port URL when status has positive port', () => {
      expect(resolveBackendBaseUrl({ port: 51234 })).toBe('http://127.0.0.1:51234');
      expect(resolveBackendBaseUrl({ port: 8080 })).toBe('http://127.0.0.1:8080');
    });

    it('BackendStatus supports optional supervisor state and stderr fields', () => {
      const status: BackendStatus = {
        running: true,
        pid: 12345,
        port: 54321,
        error: null,
        state: 'ready',
        healthy: true,
        managed: true,
        stderr_tail: ['[INFO] starting up', '[INFO] bound port 54321'],
        last_error: null,
      };
      expect(status.state).toBe('ready');
      expect(status.port).toBe(54321);
      expect(status.stderr_tail?.length).toBe(2);
      expect(resolveBackendBaseUrl(status)).toBe('http://127.0.0.1:54321');
    });
  });

  describe('checkHealth() (#14951)', () => {
    beforeEach(() => {
      vi.restoreAllMocks();
    });

    it('probes dynamic port and returns true when response is ok', async () => {
      const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200 });
      vi.stubGlobal('fetch', fetchMock);

      const status: Partial<BackendStatus> = { port: 59123 };
      const healthy = await checkHealth(status);

      expect(healthy).toBe(true);
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock).toHaveBeenCalledWith(
        'http://127.0.0.1:59123/api/health',
        expect.objectContaining({ method: 'GET' })
      );
    });

    it('returns false when health endpoint fails or network throws', async () => {
      const fetchMock = vi.fn().mockRejectedValue(new Error('Connection refused'));
      vi.stubGlobal('fetch', fetchMock);

      const healthy = await checkHealth({ port: 59123 });
      expect(healthy).toBe(false);
    });
  });
});
