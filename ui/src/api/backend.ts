/**
 * Tauri backend management API.
 *
 * Provides functions to start/stop the Python backend server and
 * retrieve diagnostic information when running inside Tauri.
 * Falls back gracefully when running in a regular browser (Vite dev).
 *
 * The default dev proxy port (8000) must match `DEFAULT_SERVER_PORT` in
 * `src/shared/python/config/typed_settings.py` (issue #6637). In Tauri the
 * supervised child may bind an ephemeral port; see `getBackendPort()`.
 */

import { useCallback, useEffect, useState } from 'react';

/** Default Python backend port for dev proxy and fallbacks (issue #6637). */
export const BACKEND_PORT = 8000;

/** Default backend base URL fallback (issue #14951). */
export const DEFAULT_BACKEND_URL =
  typeof import.meta !== 'undefined' &&
  import.meta.env &&
  import.meta.env.VITE_API_BASE_URL
    ? String(import.meta.env.VITE_API_BASE_URL)
    : `http://127.0.0.1:${BACKEND_PORT}`;

/** Port reported by the Tauri backend supervisor after start (`null` until synced). */
let tauriSupervisedPort: number | null = null;

/** Active backend port in Tauri (supervised) or the canonical default elsewhere. */
export function getBackendPort(): number {
  return tauriSupervisedPort ?? BACKEND_PORT;
}

/** Apply the port from a `BackendStatus` / diagnostics payload (#14964). */
export function applyBackendStatusPort(status: Pick<BackendStatus, 'port'>): void {
  if (!isTauri()) {
    return;
  }
  const port = status.port;
  if (typeof port === 'number' && Number.isInteger(port) && port > 0 && port <= 65535) {
    tauriSupervisedPort = port;
  }
}

/** Reset supervised port — for unit tests only. */
export function resetTauriSupervisedPortForTests(): void {
  tauriSupervisedPort = null;
}

/**
 * Returns the base URL for the Python API.
 *
 * - In Tauri mode the UI and backend are on different origins, so we
 *   return `http://localhost:<supervised-port>`.
 * - In browser/Vite mode the dev-server proxies `/api` so we return
 *   an empty string (relative URLs work fine).
 */
export function getApiBase(): string {
  if (isTauri()) {
    return `http://localhost:${getBackendPort()}`;
  }
  return '';
}

export interface BackendStatus {
  running: boolean;
  pid: number | null;
  port: number;
  error: string | null;
  state?: string;
  healthy?: boolean;
  managed?: boolean;
  /** Tail of supervised child stderr (`null` when none). */
  stderr_tail?: string | null;
  last_error?: string | null;
}

/**
 * Resolves the backend base URL given an optional BackendStatus snapshot.
 */
export function resolveBackendBaseUrl(status?: Partial<BackendStatus> | null): string {
  if (
    status &&
    typeof status.port === 'number' &&
    Number.isFinite(status.port) &&
    status.port > 0
  ) {
    applyBackendStatusPort({ port: status.port });
  }
  if (isTauri()) {
    return getApiBase();
  }
  if (
    status &&
    typeof status.port === 'number' &&
    Number.isFinite(status.port) &&
    status.port > 0
  ) {
    return `http://127.0.0.1:${status.port}`;
  }
  return DEFAULT_BACKEND_URL;
}

/**
 * Probe /api/health against the resolved backend base URL.
 */
export async function checkHealth(
  statusOrUrl?: Partial<BackendStatus> | string | null,
  timeoutMs: number = 3000
): Promise<boolean> {
  const baseUrl =
    typeof statusOrUrl === 'string'
      ? statusOrUrl.replace(/\/+$/, '')
      : resolveBackendBaseUrl(statusOrUrl);
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(`${baseUrl}/api/health`, {
      method: 'GET',
      signal: controller.signal,
    });
    clearTimeout(timer);
    return response.ok;
  } catch {
    return false;
  }
}

export interface UseBackendAutoStartResult {
  status: BackendStatus | null;
  healthy: boolean;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/** React hook to auto-start the backend and monitor health with dynamic port resolution. */
export function useBackendAutoStart(): UseBackendAutoStartResult {
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [healthy, setHealthy] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      let current = await getBackendStatus();
      if (!current.running && current.state !== 'ready' && current.state !== 'starting') {
        current = await startBackend();
      }
      setStatus(current);
      const isHealthy = await checkHealth(current);
      setHealthy(isHealthy);
      setError(current.error || current.last_error || null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setHealthy(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { status, healthy, loading, error, refresh };
}

export interface DiagnosticInfo {
  backend: BackendStatus;
  python_found: boolean;
  python_version: string | null;
  repo_root: string | null;
  local_server_found: boolean;
}

/** Check if we are running inside a Tauri window. */
export function isTauri(): boolean {
  return '__TAURI_INTERNALS__' in window;
}

async function invoke<T>(cmd: string): Promise<T> {
  const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
  return tauriInvoke<T>(cmd);
}

function syncPortFromStatus(status: BackendStatus): BackendStatus {
  applyBackendStatusPort(status);
  return status;
}

/** Start the Python backend server (Tauri only). */
export async function startBackend(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: BACKEND_PORT, error: 'Not running in Tauri' };
  }
  return syncPortFromStatus(await invoke<BackendStatus>('start_backend'));
}

/** Stop the Python backend server (Tauri only). */
export async function stopBackend(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: BACKEND_PORT, error: 'Not running in Tauri' };
  }
  return syncPortFromStatus(await invoke<BackendStatus>('stop_backend'));
}

/** Get current backend status (Tauri only). */
export async function getBackendStatus(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: BACKEND_PORT, error: null };
  }
  return syncPortFromStatus(await invoke<BackendStatus>('backend_status'));
}

/** Get comprehensive diagnostic info (Tauri only). */
export async function getDiagnostics(): Promise<DiagnosticInfo> {
  if (!isTauri()) {
    try {
      const response = await fetch(`${getApiBase()}/api/diagnostics`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      return {
        backend: {
          running: false,
          pid: null,
          port: BACKEND_PORT,
          error: error instanceof Error ? error.message : 'Unknown error',
        },
        python_found: false,
        python_version: null,
        repo_root: null,
        local_server_found: false,
      };
    }
  }
  const info = await invoke<DiagnosticInfo>('get_diagnostics');
  applyBackendStatusPort(info.backend);
  return info;
}
