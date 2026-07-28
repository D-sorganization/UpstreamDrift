/**
 * Launcher capability token used to authorise local WebSocket handshakes.
 *
 * The backend guard `enforce_local_websocket_guard` (`src/api/auth/ws_auth.py`)
 * closes any local-mode WebSocket with code 1008 unless the request carries a
 * loopback `Origin` *and* the launcher capability token issued by
 * `GET /api/launcher/manifest`.
 *
 * Historically the token was only ever populated as a side effect of the
 * Dashboard's `useLauncherManifest` hook. Deep-linking straight to
 * `/simulation` — or landing there after a failed manifest fetch — left the
 * cache empty, so `withLauncherWebSocketToken` returned a bare URL, the guard
 * rejected the handshake, and the UI reported
 * "Connection lost — restart required" the instant Start was pressed (#8077).
 *
 * `ensureLauncherCapabilityToken` closes that gap: any WebSocket caller can
 * acquire the token on demand, independently of which route mounted first.
 */

import { apiFetch } from './fetch';

const LAUNCHER_WS_TOKEN_QUERY = 'launcher_token';

/** Manifest fetch timeout, so a hung API cannot block a connect forever. */
const MANIFEST_TIMEOUT_MS = 10_000;

let launcherCapabilityToken: string | null = null;

/** In-flight manifest fetch, shared so concurrent callers issue one request. */
let inFlightTokenFetch: Promise<string | null> | null = null;

/**
 * Cache the launcher capability token for local WebSocket handshakes.
 *
 * The token is issued by `/api/launcher/manifest` and is only meaningful for
 * same-session local launcher traffic.
 */
export function setLauncherCapabilityToken(token: string | null | undefined): void {
  launcherCapabilityToken = token && token.length > 0 ? token : null;
  // A caller that explicitly sets (or clears) the token wins over any fetch
  // still in flight; drop the shared promise so the next ensure() re-reads.
  inFlightTokenFetch = null;
}

export function getLauncherCapabilityToken(): string | null {
  return launcherCapabilityToken;
}

/**
 * Return the cached launcher capability token, fetching it if absent (#8077).
 *
 * Safe to call before every WebSocket connect: it is a no-op once the token is
 * cached, and concurrent callers share a single in-flight manifest request.
 *
 * A failure here is deliberately non-fatal — the caller still attempts the
 * handshake so that deployments without the local launcher guard (cloud mode,
 * where bearer auth applies instead) keep working. The connection error, if
 * any, is then reported by the socket itself rather than pre-empted here.
 *
 * @returns The token, or `null` when it could not be obtained.
 * @postcondition On a successful fetch, `getLauncherCapabilityToken()` returns
 *   the same non-empty string.
 */
export async function ensureLauncherCapabilityToken(): Promise<string | null> {
  if (launcherCapabilityToken) {
    return launcherCapabilityToken;
  }
  if (inFlightTokenFetch) {
    return inFlightTokenFetch;
  }

  inFlightTokenFetch = (async (): Promise<string | null> => {
    try {
      const body = await apiFetch<Record<string, unknown>>('/api/launcher/manifest', {
        timeoutMs: MANIFEST_TIMEOUT_MS,
      });
      const token = body.launcher_csrf_token;
      if (typeof token === 'string' && token.length > 0) {
        launcherCapabilityToken = token;
        return token;
      }
      return null;
    } catch {
      // Network error, timeout, or non-JSON body: fall through to null and let
      // the WebSocket surface the real failure.
      return null;
    } finally {
      inFlightTokenFetch = null;
    }
  })();

  return inFlightTokenFetch;
}

export function withLauncherWebSocketToken(url: string): string {
  if (!url || url.trim().length === 0) {
    throw new Error('url must be a non-empty string');
  }
  const token = getLauncherCapabilityToken();
  if (!token) {
    return url;
  }
  const parsed = new URL(url, window.location.href);
  parsed.searchParams.set(LAUNCHER_WS_TOKEN_QUERY, token);
  return parsed.toString();
}
