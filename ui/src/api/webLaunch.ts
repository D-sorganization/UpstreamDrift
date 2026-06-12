/**
 * Web launch contract resolution (issue #7461).
 *
 * The shared launcher manifest declares per tile how it is reachable from
 * the web app (`web: {mode, route?, reason?}`). This module turns that
 * declaration plus the runtime environment (Tauri vs browser, local vs
 * remote API) into a concrete, honest launch action so the dashboard never
 * renders a button that silently does nothing or opens an invisible window
 * on a remote server.
 */

import { isTauri } from './backend';
import type { LauncherTile } from './useLauncherManifest';

/** What clicking "launch" on a tile should actually do. */
export type TileLaunchAction =
    | { kind: 'navigate'; route: string }
    | { kind: 'native-window' }
    | { kind: 'blocked'; badge: string; reason: string };

const LOCAL_HOSTNAMES = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);

/** True when the page itself is served from the local machine. */
export function isLocalHostname(
    hostname: string = window.location.hostname,
): boolean {
    return LOCAL_HOSTNAMES.has(hostname);
}

/**
 * Native-window launches spawn a Qt window on the machine running the API
 * server. That is only meaningful when that machine is the user's machine:
 * under Tauri (bundled local backend) or when the page/API is on localhost.
 */
export function canLaunchNativeWindow(hostname?: string): boolean {
    return isTauri() || isLocalHostname(hostname);
}

const DESKTOP_ONLY_REASON =
    'Opens a native desktop window on the machine running the API server. ' +
    'Use the desktop app or run the server locally.';

/**
 * Resolve a tile's launch contract into a concrete action.
 *
 * @param tile - Tile from the launcher manifest.
 * @param nativeWindowAllowed - Whether native-window launches are honest in
 *   the current environment. Defaults to runtime detection; injectable for
 *   components and tests.
 */
export function resolveTileLaunchAction(
    tile: Pick<LauncherTile, 'web'>,
    nativeWindowAllowed: boolean = canLaunchNativeWindow(),
): TileLaunchAction {
    const web = tile.web;
    if (!web) {
        // Legacy backend without a contract: preserve the native-launch
        // behavior, but apply the same locality gate so we never trigger an
        // invisible server-side window.
        return nativeWindowAllowed
            ? { kind: 'native-window' }
            : { kind: 'blocked', badge: 'Desktop app only', reason: DESKTOP_ONLY_REASON };
    }
    switch (web.mode) {
        case 'route':
            return web.route
                ? { kind: 'navigate', route: web.route }
                : {
                      kind: 'blocked',
                      badge: 'Unavailable',
                      reason: 'Tile declares route mode without a route',
                  };
        case 'native-window':
            return nativeWindowAllowed
                ? { kind: 'native-window' }
                : { kind: 'blocked', badge: 'Desktop app only', reason: DESKTOP_ONLY_REASON };
        case 'unavailable':
            return {
                kind: 'blocked',
                badge: 'Unavailable',
                reason: web.reason ?? 'Not available in the web app',
            };
    }
}
