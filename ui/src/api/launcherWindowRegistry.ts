import type { LauncherTile } from './useLauncherManifest';

export const LAUNCHER_WINDOW_REGISTRY_KEY = 'upstream-drift.launcher.windows.v1';

export interface LauncherWindowRecord {
    tileId: string;
    name: string;
    launchedAt: string;
    focusedAt: string;
    launchCount: number;
}

interface StoredRegistry {
    version: 1;
    records: LauncherWindowRecord[];
}

function parseRegistry(raw: string | null): StoredRegistry {
    if (!raw) {
        return { version: 1, records: [] };
    }

    try {
        const parsed = JSON.parse(raw) as Partial<StoredRegistry>;
        if (parsed.version !== 1 || !Array.isArray(parsed.records)) {
            return { version: 1, records: [] };
        }
        return {
            version: 1,
            records: parsed.records.filter(isWindowRecord),
        };
    } catch {
        return { version: 1, records: [] };
    }
}

function isWindowRecord(value: unknown): value is LauncherWindowRecord {
    if (!value || typeof value !== 'object') {
        return false;
    }
    const record = value as Partial<LauncherWindowRecord>;
    return (
        typeof record.tileId === 'string' &&
        record.tileId.length > 0 &&
        typeof record.name === 'string' &&
        record.name.length > 0 &&
        typeof record.launchedAt === 'string' &&
        typeof record.focusedAt === 'string' &&
        typeof record.launchCount === 'number' &&
        Number.isFinite(record.launchCount)
    );
}

function readRegistryItem(storage: Storage): string | null {
    try {
        if (typeof storage.getItem !== 'function') {
            return null;
        }
        return storage.getItem(LAUNCHER_WINDOW_REGISTRY_KEY);
    } catch {
        return null;
    }
}

function writeRegistry(storage: Storage, registry: StoredRegistry): void {
    try {
        if (typeof storage.setItem === 'function') {
            storage.setItem(LAUNCHER_WINDOW_REGISTRY_KEY, JSON.stringify(registry));
        }
    } catch {
        // Storage can be disabled in hardened desktop shells; launching should still work.
    }
}

export function loadLauncherWindowRecords(storage: Storage = window.localStorage): LauncherWindowRecord[] {
    return parseRegistry(readRegistryItem(storage)).records;
}

export function reconcileLauncherWindowRecords(
    records: LauncherWindowRecord[],
    tiles: LauncherTile[]
): LauncherWindowRecord[] {
    const liveTileIds = new Set(tiles.map((tile) => tile.id));
    return records.filter((record) => liveTileIds.has(record.tileId));
}

export function recordLauncherWindowLaunch(
    tile: LauncherTile,
    options: { storage?: Storage; now?: Date } = {}
): LauncherWindowRecord[] {
    const storage = options.storage ?? window.localStorage;
    const timestamp = (options.now ?? new Date()).toISOString();
    const records = loadLauncherWindowRecords(storage);
    const existing = records.find((record) => record.tileId === tile.id);
    const nextRecord: LauncherWindowRecord = existing
        ? {
            ...existing,
            name: tile.name,
            focusedAt: timestamp,
            launchCount: existing.launchCount + 1,
        }
        : {
            tileId: tile.id,
            name: tile.name,
            launchedAt: timestamp,
            focusedAt: timestamp,
            launchCount: 1,
        };

    const nextRecords = [
        nextRecord,
        ...records.filter((record) => record.tileId !== tile.id),
    ];
    writeRegistry(storage, { version: 1, records: nextRecords });
    return nextRecords;
}

export function persistLauncherWindowRecords(
    records: LauncherWindowRecord[],
    storage: Storage = window.localStorage
): void {
    writeRegistry(storage, { version: 1, records: records.filter(isWindowRecord) });
}
