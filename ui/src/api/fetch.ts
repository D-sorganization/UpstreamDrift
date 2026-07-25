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
 * Default request timeout (issue #8080).
 *
 * `window.fetch` has no timeout of its own: if the API accepts the connection
 * but never answers, the promise never settles and any caller awaiting it stays
 * in its loading state forever. Motion Capture sat on "Loading sources..."
 * indefinitely for exactly this reason. Every `apiFetch` call now aborts rather
 * than hanging, so callers always reach a terminal state they can render.
 */
export const DEFAULT_TIMEOUT_MS = 15_000;

/** Options accepted by `apiFetch` on top of the standard `RequestInit`. */
export interface ApiFetchInit extends RequestInit {
  /** Abort after this many ms. Defaults to `DEFAULT_TIMEOUT_MS`; 0 disables. */
  timeoutMs?: number;
}

/**
 * Build the AbortSignal for a request, combining any caller-supplied signal
 * with the timeout.
 *
 * @param timeoutMs - Timeout in ms; 0 or negative disables the timeout.
 * @param callerSignal - A signal the caller wants to keep honouring.
 * @returns The signal to pass to `fetch`, or undefined when neither applies.
 */
function buildSignal(
  timeoutMs: number,
  callerSignal: AbortSignal | null | undefined,
): AbortSignal | undefined {
  if (timeoutMs <= 0) {
    return callerSignal ?? undefined;
  }
  const timeoutSignal = AbortSignal.timeout(timeoutMs);
  if (!callerSignal) {
    return timeoutSignal;
  }
  // `AbortSignal.any` is available in every browser the app targets; fall back
  // to the timeout alone if a test environment lacks it.
  if (typeof AbortSignal.any === 'function') {
    return AbortSignal.any([callerSignal, timeoutSignal]);
  }
  return timeoutSignal;
}

/**
 * apiFetch<T> — typed HTTP helper with consistent error handling.
 *
 * @param path - API path starting with `/`, e.g. `/api/engines`
 * @param init - Optional `RequestInit` options plus `timeoutMs` (#8080)
 * @returns Parsed JSON body typed as `T`
 * @throws Error with a human-readable message on HTTP, network, or timeout
 *   errors. A timeout message always contains the word "timed out" so callers
 *   can distinguish it from a refused connection.
 */
export async function apiFetch<T>(
  path: string,
  init?: ApiFetchInit,
): Promise<T> {
  const url = `${getApiBase()}${path}`;
  const { timeoutMs = DEFAULT_TIMEOUT_MS, ...requestInit } = init ?? {};

  const mergedInit: RequestInit = {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...requestInit,
    signal: buildSignal(timeoutMs, requestInit.signal),
  };

  let response: Response;
  try {
    response = await fetch(url, mergedInit);
  } catch (err) {
    if (err instanceof DOMException && err.name === 'TimeoutError') {
      throw new Error(`Request timed out after ${timeoutMs}ms — ${path}`);
    }
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw new Error(`Request aborted — ${path}`);
    }
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
 * apiFetchParsed — `apiFetch` plus runtime validation of the response body
 * (issue #7165).
 *
 * `apiFetch<T>` only asserts the response type at compile time; a backend on a
 * different version can return a malformed shape that is stored as-is and later
 * crashes deep in a component render. This wrapper runs the body through a
 * caller-supplied `parse` function so an invalid payload surfaces as a thrown
 * error (which hooks turn into their `error` state) rather than a render-time
 * `TypeError`.
 *
 * @param path - API path starting with `/`
 * @param parse - Validator that returns the parsed value or throws on a bad shape
 * @param init - Optional `RequestInit`
 * @returns The validated value
 * @throws Error on HTTP/network failure or when `parse` rejects the body
 */
export async function apiFetchParsed<T>(
  path: string,
  parse: (raw: unknown) => T,
  init?: RequestInit,
): Promise<T> {
  const raw = await apiFetch<unknown>(path, init);
  return parse(raw);
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
