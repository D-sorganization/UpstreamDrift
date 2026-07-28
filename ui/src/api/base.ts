/**
 * Dependency-free API base URL helpers.
 *
 * Keep this module free of imports from higher-level API helpers so request
 * wrappers and backend lifecycle code can both depend on it without cycles.
 */

/** Single source of truth for the Python backend port (issue #6637). */
export const BACKEND_PORT = 8000;

/** Check if we are running inside a Tauri window. */
export function isTauri(): boolean {
  return '__TAURI_INTERNALS__' in window;
}

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
