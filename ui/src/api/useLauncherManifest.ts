/**
 * Launcher Manifest Hook — Fetches tile definitions from the shared manifest API.
 *
 * This hook reads from `/api/launcher/manifest` which is served by the Python
 * backend and derived from `launcher_manifest.json` — the single source of truth
 * for both PyQt and Tauri launchers.
 *
 * Design by Contract:
 *   Postcondition: returned tiles are always sorted by `order` field
 *   Invariant: tile IDs are unique
 */

import { useState, useEffect, useCallback } from 'react';

export interface LauncherTile {
    id: string;
    name: string;
    description: string;
    category: 'physics_engine' | 'tool' | 'external';
    type: string;
    path: string;
    logo: string;
    status: string;
    capabilities: string[];
    order: number;
    engine_type?: string;
    hidden?: boolean;
}

export interface LauncherManifest {
    version: string;
    description: string;
    tiles: LauncherTile[];
    launcher_csrf_token?: string;
    launcher_csrf_header?: string;
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
            const response = await fetch('/api/launcher/manifest');
            if (!response.ok) {
                throw new Error(`Failed to fetch manifest: ${response.status}`);
            }
            const data: LauncherManifest = await response.json();

            if (!Array.isArray(data.tiles)) {
                throw new Error('Invalid manifest: tiles must be an array');
            }

            // Filter hidden tiles (e.g. legacy aliases retained for saved
            // layout resolution) so the dashboard does not render duplicate
            // cards for the same app (issue #4507).
            data.tiles = data.tiles.filter((tile) => !tile.hidden);

            // DBC Postcondition: sort tiles by order
            data.tiles.sort((a, b) => a.order - b.order);

            setManifest(data);
            setLoadState('loaded');
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to load launcher manifest';
            setError(message);
            setLoadState('error');
        }
    }, []);

    useEffect(() => {
        fetchManifest();
    }, [fetchManifest]);

    const tiles = manifest?.tiles ?? [];
    const engines = tiles.filter((t) => t.category === 'physics_engine');
    const tools = tiles.filter((t) => t.category === 'tool' || t.category === 'external');
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
