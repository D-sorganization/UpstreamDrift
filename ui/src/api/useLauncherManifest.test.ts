/**
 * TDD Tests for useLauncherManifest hook.
 *
 * Tests the manifest fetching, parsing, and category filtering.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useLauncherManifest } from './useLauncherManifest';
import type { LauncherManifest } from './useLauncherManifest';

const MOCK_MANIFEST: LauncherManifest = {
    version: '1.0.0',
    description: 'Test manifest',
    tiles: [
        {
            id: 'model_explorer',
            name: 'Model Explorer',
            description: 'Browse models',
            category: 'tool',
            type: 'special_app',
            path: 'src/tools/urdf_generator/launch_urdf_generator.py',
            logo: 'urdf_icon.png',
            status: 'utility',
            capabilities: ['model_browsing'],
            order: 1,
        },
        {
            id: 'mujoco_unified',
            name: 'MuJoCo',
            description: 'MuJoCo simulation',
            category: 'physics_engine',
            type: 'custom_humanoid',
            path: 'src/launchers/mujoco_unified_launcher.py',
            logo: 'mujoco_humanoid.png',
            status: 'gui_ready',
            capabilities: ['rigid_body'],
            order: 2,
            engine_type: 'mujoco',
        },
        {
            id: 'drake_golf',
            name: 'Drake',
            description: 'Drake dynamics',
            category: 'physics_engine',
            type: 'drake',
            path: 'src/engines/drake.py',
            logo: 'drake.png',
            status: 'gui_ready',
            capabilities: ['rigid_body'],
            order: 3,
            engine_type: 'drake',
        },
        {
            id: 'matlab_unified',
            name: 'Matlab Models',
            description: 'Matlab stuff',
            category: 'external',
            type: 'special_app',
            path: 'src/launchers/matlab.py',
            logo: 'matlab_logo.png',
            status: 'external',
            capabilities: ['simscape'],
            order: 8,
        },
    ],
};

// Save original fetch
const originalFetch = global.fetch;

describe('useLauncherManifest', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    afterEach(() => {
        global.fetch = originalFetch;
    });

    it('starts in idle state and transitions to loaded', async () => {
        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(MOCK_MANIFEST),
        });

        const { result } = renderHook(() => useLauncherManifest());

        // Initially loading
        expect(result.current.loadState).toBe('loading');

        await waitFor(() => {
            expect(result.current.loadState).toBe('loaded');
        });

        expect(result.current.manifest).not.toBeNull();
        expect(result.current.tiles).toHaveLength(4);
    });

    it('returns tiles sorted by order', async () => {
        // Shuffle tiles to test sorting
        const shuffled = {
            ...MOCK_MANIFEST,
            tiles: [...MOCK_MANIFEST.tiles].reverse(),
        };

        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(shuffled),
        });

        const { result } = renderHook(() => useLauncherManifest());

        await waitFor(() => {
            expect(result.current.loadState).toBe('loaded');
        });

        expect(result.current.tiles[0].id).toBe('model_explorer');
        expect(result.current.tiles[0].order).toBe(1);
    });

    it('correctly filters engines', async () => {
        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(MOCK_MANIFEST),
        });

        const { result } = renderHook(() => useLauncherManifest());

        await waitFor(() => {
            expect(result.current.loadState).toBe('loaded');
        });

        expect(result.current.engines).toHaveLength(2);
        expect(result.current.engines.every((e) => e.category === 'physics_engine')).toBe(true);
    });

    it('correctly filters tools and external tiles', async () => {
        global.fetch = vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(MOCK_MANIFEST),
        });

        const { result } = renderHook(() => useLauncherManifest());

        await waitFor(() => {
            expect(result.current.loadState).toBe('loaded');
        });

        expect(result.current.tools).toHaveLength(2);
        expect(result.current.tools.some((t) => t.id === 'model_explorer')).toBe(true);
        expect(result.current.tools.some((t) => t.id === 'matlab_unified')).toBe(true);
    });

    it('transitions to error state on fetch failure', async () => {
        global.fetch = vi.fn().mockResolvedValue({
            ok: false,
            status: 500,
        });

        const { result } = renderHook(() => useLauncherManifest());

        await waitFor(() => {
            expect(result.current.loadState).toBe('error');
        });

        expect(result.current.error).toContain('500');
        expect(result.current.tiles).toHaveLength(0);
    });

    it('transitions to error state on network failure', async () => {
        global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

        const { result } = renderHook(() => useLauncherManifest());

        await waitFor(() => {
            expect(result.current.loadState).toBe('error');
        });

        expect(result.current.error).toContain('Network error');
    });

    it('fetches from /api/launcher/manifest', async () => {
        const mockFetch = vi.fn().mockResolvedValue({
            ok: true,
            json: () => Promise.resolve(MOCK_MANIFEST),
        });
        global.fetch = mockFetch;

        const { result } = renderHook(() => useLauncherManifest());

        await waitFor(() => {
            expect(result.current.loadState).toBe('loaded');
        });

        expect(mockFetch).toHaveBeenCalledWith('/api/launcher/manifest');
    });
});
