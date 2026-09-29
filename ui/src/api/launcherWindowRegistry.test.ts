import { describe, expect, it, beforeEach } from 'vitest';
import type { LauncherTile } from './useLauncherManifest';
import {
    LAUNCHER_WINDOW_REGISTRY_KEY,
    loadLauncherWindowRecords,
    persistLauncherWindowRecords,
    reconcileLauncherWindowRecords,
    recordLauncherWindowLaunch,
} from './launcherWindowRegistry';

function installMemoryStorage(): void {
    const values = new Map<string, string>();
    Object.defineProperty(window, 'localStorage', {
        configurable: true,
        value: {
            getItem: (key: string) => values.get(key) ?? null,
            setItem: (key: string, value: string) => {
                values.set(key, value);
            },
            removeItem: (key: string) => {
                values.delete(key);
            },
            clear: () => values.clear(),
        },
    });
}

const tile = (id: string, name: string): LauncherTile => ({
    id,
    name,
    description: `${name} description`,
    category: 'tool',
    type: 'special_app',
    path: `src/tools/${id}.py`,
    logo: `${id}.png`,
    status: 'utility',
    capabilities: [],
    order: 1,
    default_launch: 'tab',
});

describe('launcherWindowRegistry', () => {
    beforeEach(() => {
        installMemoryStorage();
        window.localStorage.removeItem(LAUNCHER_WINDOW_REGISTRY_KEY);
    });

    it('persists launched tiles newest first', () => {
        const first = tile('model_explorer', 'Model Explorer');
        const second = tile('motion_capture', 'Motion Capture');

        recordLauncherWindowLaunch(first, { now: new Date('2026-06-10T12:00:00Z') });
        recordLauncherWindowLaunch(second, { now: new Date('2026-06-10T12:05:00Z') });

        const records = loadLauncherWindowRecords();
        expect(records.map((record) => record.tileId)).toEqual(['motion_capture', 'model_explorer']);
        expect(records[0]).toMatchObject({
            name: 'Motion Capture',
            launchCount: 1,
            launchedAt: '2026-06-10T12:05:00.000Z',
            focusedAt: '2026-06-10T12:05:00.000Z',
        });
    });

    it('updates focus time and launch count when a tile is relaunched', () => {
        const modelExplorer = tile('model_explorer', 'Model Explorer');

        recordLauncherWindowLaunch(modelExplorer, { now: new Date('2026-06-10T12:00:00Z') });
        recordLauncherWindowLaunch(modelExplorer, { now: new Date('2026-06-10T12:10:00Z') });

        expect(loadLauncherWindowRecords()).toEqual([
            {
                tileId: 'model_explorer',
                name: 'Model Explorer',
                launchedAt: '2026-06-10T12:00:00.000Z',
                focusedAt: '2026-06-10T12:10:00.000Z',
                launchCount: 2,
            },
        ]);
    });

    it('drops malformed storage instead of crashing dashboard startup', () => {
        window.localStorage.setItem(LAUNCHER_WINDOW_REGISTRY_KEY, '{bad json');

        expect(loadLauncherWindowRecords()).toEqual([]);
    });

    it('reconciles persisted window records with the manifest tile ids', () => {
        persistLauncherWindowRecords([
            {
                tileId: 'model_explorer',
                name: 'Model Explorer',
                launchedAt: '2026-06-10T12:00:00.000Z',
                focusedAt: '2026-06-10T12:00:00.000Z',
                launchCount: 1,
            },
            {
                tileId: 'removed_tile',
                name: 'Removed',
                launchedAt: '2026-06-10T12:01:00.000Z',
                focusedAt: '2026-06-10T12:01:00.000Z',
                launchCount: 1,
            },
        ]);

        const records = reconcileLauncherWindowRecords(loadLauncherWindowRecords(), [
            tile('model_explorer', 'Model Explorer'),
        ]);

        expect(records.map((record) => record.tileId)).toEqual(['model_explorer']);
    });
});
