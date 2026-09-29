/**
 * Vite dev-server proxy contract (issues #8076, #8077).
 *
 * Kept in its own dependency-free module so it can be unit tested without
 * importing `vite.config.ts` — that pulls in esbuild, which refuses to load
 * under the jsdom test environment.
 *
 * The defect fixed here is #8076: the proxy targeted port 8001 while
 * `BACKEND_PORT` in `src/api/backend.ts`, the `--port` default of
 * `launch_upstream_drift.py`, `ui/README.md` and `docs/README.md` all said
 * 8000. Following the documented startup path left the dashboard on
 * `HTTP 500 — /api/launcher/manifest` until the user discovered and started a
 * second API process on the undocumented port.
 *
 * `/api/ws` is additionally declared before `/api`, and both entries set
 * `ws: true`. Vite 7 was measured to upgrade correctly even with the old
 * ordering, so this is hygiene, not a bug fix — it removes a silent
 * dependency on undocumented first-match behaviour. The WebSocket failure
 * reported in #8077 had a different cause (a missing launcher capability
 * token); see `src/api/websocketToken.ts`.
 */

/** Vite dev-server port. Documented in `ui/README.md`. */
export const DEV_SERVER_PORT = 5180;

/**
 * Canonical Python API port for local development.
 *
 * MUST equal `BACKEND_PORT` in `ui/src/api/backend.ts` — the frontend's
 * declared single source of truth — which in turn matches `BACKEND_PORT` in
 * `ui/src-tauri/src/lib.rs` and `DEFAULT_SERVER_PORT` in
 * `src/shared/python/config/typed_settings.py`.
 */
export const DEFAULT_API_PORT = 8000;

/** Minimal structural type for one Vite proxy entry (avoids a vite import). */
export interface DevProxyEntry {
  target: string;
  changeOrigin: boolean;
  ws: boolean;
}

/**
 * Resolve the API port from the environment.
 *
 * @param raw - Value of `VITE_API_PORT`, or undefined when unset.
 * @returns The port to proxy to; `DEFAULT_API_PORT` when unset or blank.
 * @throws Error when `raw` is set but is not an integer in 1..65535.
 */
export function resolveApiPort(raw: string | undefined): number {
  if (raw === undefined || raw.trim() === '') {
    return DEFAULT_API_PORT;
  }
  const parsed = Number.parseInt(raw.trim(), 10);
  if (
    !Number.isInteger(parsed) ||
    String(parsed) !== raw.trim() ||
    parsed < 1 ||
    parsed > 65535
  ) {
    throw new Error(
      `VITE_API_PORT must be an integer in 1..65535, received "${raw}"`,
    );
  }
  return parsed;
}

/**
 * Build the dev-server proxy table.
 *
 * Key order is load-bearing: `/api/ws` must come first so WebSocket upgrades
 * are not shadowed by the plain-HTTP `/api` entry (#8077).
 *
 * @param apiPort - Port the Python API listens on.
 * @returns Proxy table in the order Vite will evaluate it.
 * @throws Error when `apiPort` is not a valid TCP port.
 */
export function buildDevProxy(
  apiPort: number = DEFAULT_API_PORT,
): Record<string, DevProxyEntry> {
  if (!Number.isInteger(apiPort) || apiPort < 1 || apiPort > 65535) {
    throw new Error(`apiPort must be an integer in 1..65535, received ${apiPort}`);
  }
  return {
    '/api/ws': {
      target: `ws://localhost:${apiPort}`,
      changeOrigin: true,
      ws: true,
    },
    '/api': {
      target: `http://localhost:${apiPort}`,
      changeOrigin: true,
      // Defence in depth: a future WebSocket path that stops matching
      // `/api/ws` still upgrades instead of silently failing.
      ws: true,
    },
    // The vendored Impact Explorer bundle is served by the Python API
    // (`_mount_impact_explorer_directory`), not by Vite. Without this entry
    // the dev server answers the page's availability probe with its own SPA
    // fallback (HTTP 200), so /tools/impact-explorer would embed a broken
    // frame in dev while working in production — the exact silent divergence
    // the probe exists to prevent.
    '/impact-explorer-app': {
      target: `http://localhost:${apiPort}`,
      changeOrigin: true,
      // Static bundle needs no upgrades, but the proxy contract test holds
      // every entry to the same defence-in-depth posture as /api.
      ws: true,
    },
  };
}
