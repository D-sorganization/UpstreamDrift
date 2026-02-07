/**
 * Tauri backend management API.
 *
 * Provides functions to start/stop the Python backend server and
 * retrieve diagnostic information when running inside Tauri.
 * Falls back gracefully when running in a regular browser (Vite dev).
 */

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
    return { running: false, pid: null, port: 8080, error: 'Not running in Tauri' };
  }
  return invoke<BackendStatus>('start_backend');
}

/** Stop the Python backend server (Tauri only). */
export async function stopBackend(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: 8080, error: 'Not running in Tauri' };
  }
  return invoke<BackendStatus>('stop_backend');
}

/** Get current backend status (Tauri only). */
export async function getBackendStatus(): Promise<BackendStatus> {
  if (!isTauri()) {
    return { running: false, pid: null, port: 8080, error: null };
  }
  return invoke<BackendStatus>('backend_status');
}

/** Get comprehensive diagnostic info (Tauri only). */
export async function getDiagnostics(): Promise<DiagnosticInfo> {
  if (!isTauri()) {
    return {
      backend: { running: false, pid: null, port: 8080, error: null },
      python_found: false,
      python_version: null,
      repo_root: null,
      local_server_found: false,
    };
  }
  return invoke<DiagnosticInfo>('get_diagnostics');
}
