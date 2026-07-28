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

import { apiFetch } from './fetch';
import { BACKEND_PORT, isTauri } from './base';

export { BACKEND_PORT, getApiBase, isTauri } from './base';

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
    try {
      return await apiFetch<DiagnosticInfo>('/api/diagnostics');
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


