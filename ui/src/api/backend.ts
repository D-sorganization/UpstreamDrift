/**
 * Tauri backend management API.
 *
 * Provides functions to start/stop the Python backend server and
 * retrieve diagnostic information when running inside Tauri.
 * Falls back gracefully when running in a regular browser (Vite dev).
 *
 * The canonical backend port (8000) must match:
 *   - `BACKEND_PORT` in `ui/src-tauri/src/lib.rs`
 *   - `DEFAULT_SERVER_PORT` in `src/shared/python/config/typed_settings.py`
 * See issue #6637.
 */

import { useState, useEffect, useCallback } from 'react';

/** Single source of truth for the Python backend port (issue #6637). */
export const BACKEND_PORT = 8000;

/** Default backend base URL fallback (issue #14951). */
export const DEFAULT_BACKEND_URL =
  typeof import.meta !== 'undefined' &&
  import.meta.env &&
  import.meta.env.VITE_API_BASE_URL
    ? String(import.meta.env.VITE_API_BASE_URL)
    : `http://127.0.0.1:${BACKEND_PORT}`;

/**
 * Returns the base URL for the Python API.
 *
 * - In Tauri mode the UI and backend are on different origins, so we
 *   return `http://localhost:BACKEND_PORT`.
 * - In browser/Vite mode the dev-server proxies `/api` so we return
 *   an empty string (relative URLs work fine).
 */
export function getApiBase(): string {
  if (isTauri()) {
    return `http://localhost:${BACKEND_PORT}`;
  }
  return '';
}

export interface BackendStatus {
  running: boolean;
  pid: number | null;
  port: number | null;
  error: string | null;
  state?: string;
  healthy?: boolean;
  managed?: boolean;
  stderr_tail?: string[];
  last_error?: string | null;
}

/**
 * Resolves the backend base URL given an optional BackendStatus snapshot.
 *
 * If a valid positive port is present in the status, returns `http://127.0.0.1:${status.port}`.
 * Otherwise falls back to DEFAULT_BACKEND_URL.
 */
export function resolveBackendBaseUrl(status?: Partial<BackendStatus> | null): string {
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
 *
 * Resolves the target base URL using `resolveBackendBaseUrl(status)` (or accepts
 * an explicit URL string), and returns true if the endpoint returns HTTP 200 OK.
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

/**
 * React hook to auto-start the backend and monitor health with dynamic port resolution.
 */
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
  // Dynamically import Tauri API only when available
  const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
  return tauriInvoke<T>(cmd);
}

/** Start the Python backend server (Tauri only). */
export async function startBackend(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: BACKEND_PORT, error: 'Not running in Tauri' };
  }
  return invoke<BackendStatus>('start_backend');
}

/** Stop the Python backend server (Tauri only). */
export async function stopBackend(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: BACKEND_PORT, error: 'Not running in Tauri' };
  }
  return invoke<BackendStatus>('stop_backend');
}

/** Get current backend status (Tauri only). */
export async function getBackendStatus(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: BACKEND_PORT, error: null };
  }
  return invoke<BackendStatus>('backend_status');
}

/** Get comprehensive diagnostic info (Tauri only). */
export async function getDiagnostics(): Promise<DiagnosticInfo> {
  if (!isTauri()) {
    // In browser mode, call the backend API endpoint.
    // NOTE: cannot use apiFetch here — fetch.ts imports getApiBase from this
    // module, so importing apiFetch back would create a circular dependency.
    // We build the URL via getApiBase() directly to stay Tauri-safe (#6897).
    try {
      const response = await fetch(`${getApiBase()}/api/diagnostics`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      // Fallback if backend is not available
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
  return invoke<DiagnosticInfo>('get_diagnostics');
}


