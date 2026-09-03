/**
 * Launcher Manifest Hook — Fetches tile definitions from the shared manifest API.
 *
 * This hook reads from `/api/launcher/manifest` which is served by the Python
 * backend and derived from `src/config/models.yaml` — the single tile registry (#9412)
 * for both PyQt and Tauri launchers.
 *
 * Design by Contract:
 *   Postcondition: returned tiles are always sorted by `order` field
 *   Invariant: tile IDs are unique
 */

import { useState, useEffect, useCallback } from 'react';
import { apiFetchParsed } from './fetch';
import { parseLauncherManifest } from './schemas';
import { setLauncherCapabilityToken } from './websocketToken';
import type {
    LauncherManifestResponse,
    LauncherTileResponse,
} from './generated/types';

/**
 * Launcher manifest payloads — generated from the API contract (issue #7447).
 *
 * Do NOT hand-write these shapes: they mirror `LauncherManifestResponse` /
 * `LauncherTileResponse` in `src/api/models/responses.py` via
 * `ui/src/api/generated/types.ts`.
 */
export type LauncherTile = LauncherTileResponse;
export type LauncherCategory = LauncherTile['category'];
export type LauncherManifest = LauncherManifestResponse;

/** How a tile is reachable from the web app (issue #7461). */
export type WebLaunchMode = 'route' | 'native-window' | 'unavailable';

/**
 * Web launch contract declared per tile in the shared manifest.
 *
 * - `route`: in-app React route navigation (`route` required, starts with "/")
 * - `native-window`: spawns a Qt window on the API server's machine — only
 *   honest when running under Tauri or against a localhost API
 * - `unavailable`: no web affordance; `reason` explains why
 */
export interface WebLaunchContract {
    mode: WebLaunchMode;
    route?: string;
    reason?: string;
}

export type ManifestLoadState = 'idle' | 'loading' | 'loaded' | 'error';

interface UseLauncherManifestResult {
    manifest: LauncherManifest | null;
    tiles: LauncherTile[];
    engines: LauncherTile[];
    tools: LauncherTile[];
    launcherCsrfToken: string | null;
    launcherCsrfHeader: string;
    loadState: ManifestLoadState;
    error: string | null;
    refetch: () => Promise<void>;
}

export function useLauncherManifest(): UseLauncherManifestResult {
    const [manifest, setManifest] = useState<LauncherManifest | null>(null);
    const [loadState, setLoadState] = useState<ManifestLoadState>('idle');
    const [error, setError] = useState<string | null>(null);

    const fetchManifest = useCallback(async () => {
        setLoadState('loading');
        setError(null);

        try {
            // Runtime-validate the payload (issue #7165): tiles must be an
            // array and every tile must carry a finite-integer `order`, else
            // the sort below produces NaN comparisons and silently-wrong order.
            const data = await apiFetchParsed(
                '/api/launcher/manifest',
                parseLauncherManifest,
            );

            // Filter hidden tiles (e.g. legacy aliases retained for saved
            // layout resolution) so the dashboard does not render duplicate
            // cards for the same app (issue #4507).
            data.tiles = data.tiles.filter((tile) => !tile.hidden);

            // DBC Postcondition: sort tiles by order
            data.tiles.sort((a, b) => a.order - b.order);

            setManifest(data);
            setLauncherCapabilityToken(data.launcher_csrf_token ?? null);
            setLoadState('loaded');
        } catch (err) {
            setLauncherCapabilityToken(null);
            const message = err instanceof Error ? err.message : 'Failed to load launcher manifest';
            setError(message);
            setLoadState('error');
        }
    }, []);

    useEffect(() => {
        // Defer to a microtask so the synchronous `setLoadState('loading')`
        // inside fetchManifest does not run in the effect body (mirrors
        // useEngineCapabilities; satisfies react-hooks/set-state-in-effect).
        void Promise.resolve().then(() => {
            fetchManifest();
        });
    }, [fetchManifest]);

    const tiles = manifest?.tiles ?? [];
    const engines = tiles.filter((t) => t.category === 'physics_engine');
    const tools = tiles.filter((t) => t.category !== 'physics_engine');
    const launcherCsrfToken = manifest?.launcher_csrf_token ?? null;
    const launcherCsrfHeader = manifest?.launcher_csrf_header ?? 'X-Launcher-CSRF-Token';

    return {
        manifest,
        tiles,
        engines,
        tools,
        launcherCsrfToken,
        launcherCsrfHeader,
        loadState,
        error,
        refetch: fetchManifest,
    };
}
