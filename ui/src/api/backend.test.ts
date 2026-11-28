import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}));

import { invoke } from '@tauri-apps/api/core';
import {
  BACKEND_PORT,
  DEFAULT_BACKEND_URL,
  applyBackendStatusPort,
  checkHealth,
  getApiBase,
  getBackendPort,
  getBackendStatus,
  isTauri,
  resetTauriSupervisedPortForTests,
  resolveBackendBaseUrl,
  startBackend,
  type BackendStatus,
} from './backend';

/**
 * Tests for the backend helper module (issues #6637, #14951, #14964).
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
      resetTauriSupervisedPortForTests();
      if ('__TAURI_INTERNALS__' in window) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        delete (window as any).__TAURI_INTERNALS__;
      }
    });

    it('returns empty string in browser (non-Tauri) mode', () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (window as any).__TAURI_INTERNALS__;
      expect(isTauri()).toBe(false);
      expect(getApiBase()).toBe('');
    });

    it('returns http://localhost:BACKEND_PORT in Tauri mode before sync', () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__TAURI_INTERNALS__ = {};
      expect(isTauri()).toBe(true);
      expect(getApiBase()).toBe(`http://localhost:${BACKEND_PORT}`);
    });

    it('uses the supervised port after BackendStatus sync (#14964)', () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__TAURI_INTERNALS__ = {};
      const supervised = 38427;
      applyBackendStatusPort({ port: supervised });
      expect(getBackendPort()).toBe(supervised);
      expect(getApiBase()).toBe(`http://localhost:${supervised}`);
    });
  });

  describe('dynamic backend port resolution (#14951)', () => {
    afterEach(() => {
      resetTauriSupervisedPortForTests();
    });

    it('defaults DEFAULT_BACKEND_URL to 127.0.0.1:8000', () => {
      expect(DEFAULT_BACKEND_URL).toBe('http://127.0.0.1:8000');
    });

    it('resolveBackendBaseUrl returns DEFAULT_BACKEND_URL when status is undefined or null', () => {
      expect(resolveBackendBaseUrl()).toBe('http://127.0.0.1:8000');
      expect(resolveBackendBaseUrl(null)).toBe('http://127.0.0.1:8000');
    });

    it('resolveBackendBaseUrl returns DEFAULT_BACKEND_URL when port is 0 or negative', () => {
      expect(resolveBackendBaseUrl({ port: 0 })).toBe('http://127.0.0.1:8000');
      expect(resolveBackendBaseUrl({ port: -1 })).toBe('http://127.0.0.1:8000');
    });

    it('resolveBackendBaseUrl returns dynamic port URL when status has positive port', () => {
      expect(resolveBackendBaseUrl({ port: 51234 })).toBe('http://127.0.0.1:51234');
      expect(resolveBackendBaseUrl({ port: 8080 })).toBe('http://127.0.0.1:8080');
    });

    it('BackendStatus supports optional supervisor state and stderr string wire', () => {
      const status: BackendStatus = {
        running: true,
        pid: 12345,
        port: 54321,
        error: null,
        state: 'ready',
        healthy: true,
        managed: true,
        stderr_tail: '[INFO] starting up\n[INFO] bound port 54321',
        last_error: null,
      };
      expect(status.state).toBe('ready');
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

      const healthy = await checkHealth({ port: 59123 });

      expect(healthy).toBe(true);
      expect(fetchMock).toHaveBeenCalledWith(
        'http://127.0.0.1:59123/api/health',
        expect.objectContaining({ method: 'GET' })
      );
    });

    it('returns false when health endpoint fails or network throws', async () => {
      vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('Connection refused')));
      expect(await checkHealth({ port: 59123 })).toBe(false);
    });
  });

  describe('Tauri supervised port IPC (#14964)', () => {
    beforeEach(() => {
      vi.clearAllMocks();
      resetTauriSupervisedPortForTests();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (window as any).__TAURI_INTERNALS__ = {};
    });

    afterEach(() => {
      resetTauriSupervisedPortForTests();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (window as any).__TAURI_INTERNALS__;
    });

    it('first backend_status sync drives HTTP base before any fetch', async () => {
      vi.mocked(invoke).mockResolvedValueOnce({
        running: false,
        pid: 123,
        port: 45123,
        state: 'starting',
        error: null,
        stderr_tail: null,
      });
      await getBackendStatus();
      expect(getApiBase()).toBe('http://localhost:45123');
    });

    it('restart reports a new supervised port for HTTP and WebSocket', async () => {
      vi.mocked(invoke).mockResolvedValueOnce({
        running: true,
        pid: 1,
        port: 45123,
        error: null,
      });
      await startBackend();
      vi.mocked(invoke).mockResolvedValueOnce({
        running: true,
        pid: 2,
        port: 55234,
        error: null,
      });
      await getBackendStatus();
      expect(getApiBase()).toBe('http://localhost:55234');
      expect(getApiBase().replace(/^http/, 'ws')).toBe('ws://localhost:55234');
    });
  });
});
