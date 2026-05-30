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

/** Single source of truth for the Python backend port (issue #6637). */
export const BACKEND_PORT = 8000;

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
  port: number;
  error: string | null;
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
    // In browser mode, call the backend API endpoint
    try {
      const response = await fetch('/api/diagnostics');
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
