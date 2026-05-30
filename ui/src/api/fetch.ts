/**
 * Shared fetch helper — issue #6642 F8
 *
 * Provides a single, typed wrapper around `window.fetch` that:
 *  - Builds the full URL via `getApiBase()` so Tauri and browser builds
 *    both resolve to the right origin (fixes #6637 independently here too).
 *  - Throws a descriptive `Error` on non-2xx responses, extracting the
 *    `detail` field from FastAPI JSON error bodies when available.
 *  - Returns the parsed JSON body typed as `T`.
 *
 * Usage:
 *   const engines = await apiFetch<EngineStatus[]>('/api/engines');
 *   const result  = await apiFetch<GenerateResult>('/api/dataset/generate', {
 *     method: 'POST',
 *     body: JSON.stringify(params),
 *   });
 */

import { getApiBase } from './backend';

/**
 * apiFetch<T> — typed HTTP helper with consistent error handling.
 *
 * @param path - API path starting with `/`, e.g. `/api/engines`
 * @param init - Optional `RequestInit` options (method, body, headers…)
 * @returns Parsed JSON body typed as `T`
 * @throws Error with a human-readable message on HTTP or network errors
 */
export async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${getApiBase()}${path}`;

  const mergedInit: RequestInit = {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  };

  let response: Response;
  try {
    response = await fetch(url, mergedInit);
  } catch (err) {
    // Network-level error (no response at all)
    throw new Error(
      err instanceof Error ? err.message : `Network error for ${path}`,
    );
  }

  if (!response.ok) {
    // Attempt to extract a FastAPI-style `detail` field
    let detail: string | undefined;
    try {
      const body = (await response.json()) as Record<string, unknown>;
      if (typeof body.detail === 'string') {
        detail = body.detail;
      }
    } catch {
      // Body was not valid JSON — that's fine, fall through to generic message
    }
    throw new Error(detail ?? `HTTP ${response.status} ${response.statusText} — ${path}`);
  }

  return response.json() as Promise<T>;
}

/**
 * apiFetchForm — same as apiFetch but for multipart/form-data uploads.
 *
 * Do NOT set Content-Type header; the browser sets it with the boundary.
 *
 * @param path - API path
 * @param formData - FormData payload
 * @returns Parsed JSON body typed as `T`
 */
export async function apiFetchForm<T>(path: string, formData: FormData): Promise<T> {
  const url = `${getApiBase()}${path}`;

  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', body: formData });
  } catch (err) {
    throw new Error(
      err instanceof Error ? err.message : `Network error for ${path}`,
    );
  }

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = (await response.json()) as Record<string, unknown>;
      if (typeof body.detail === 'string') {
        detail = body.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(detail ?? `HTTP ${response.status} ${response.statusText} — ${path}`);
  }

  return response.json() as Promise<T>;
}
