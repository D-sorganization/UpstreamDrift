/**
 * Launcher Reachability Evaluation and Matrix Generator (ORG-04, Issue #10513).
 *
 * Evaluates whether launcher tiles can be reached from browser and Tauri contexts:
 * - Validates route existence against the canonical React route table.
 * - Distinguishes between in-app route navigation, native-window launches, and honest unavailable states.
 * - Generates an authoritative reachability matrix for all effective manifest tiles.
 */

import type { LauncherTile, WebLaunchMode } from './useLauncherManifest';
import { resolveTileLaunchAction, isLocalHostname } from './webLaunch';
import type { TileLaunchAction } from './webLaunch';

export type ReachabilityContext = 'browser' | 'tauri';

export interface ReachabilityEvaluation {
    action: TileLaunchAction;
    isReachable: boolean;
    reason?: string;
    effectiveDestination: 'route' | 'native-window' | 'unavailable';
}

export interface ReachabilityMatrixEntry {
    tileId: string;
    name: string;
    declaredMode: WebLaunchMode | string;
    route?: string | null;
    browser: ReachabilityEvaluation;
    tauri: ReachabilityEvaluation;
}

/**
 * Evaluate reachability of a tile in a specific environment context.
 */
export function evaluateTileReachability(
    tile: Pick<LauncherTile, 'id' | 'name' | 'web'>,
    context: ReachabilityContext,
    options: { browserHostname?: string } = {},
): ReachabilityEvaluation {
    const isTauriEnv = context === 'tauri';
    const isAllowed = isTauriEnv || isLocalHostname(options.browserHostname);
    const action = resolveTileLaunchAction(tile, isAllowed);

    if (action.kind === 'navigate') {
        return {
            action,
            isReachable: true,
            effectiveDestination: 'route',
        };
    }
    if (action.kind === 'native-window') {
        return {
            action,
            isReachable: true,
            effectiveDestination: 'native-window',
        };
    }
    return {
        action,
        isReachable: false,
        reason: action.reason,
        effectiveDestination: 'unavailable',
    };
}

/**
 * Generate a complete reachability matrix covering all manifest tiles in both browser and Tauri.
 */
export function generateReachabilityMatrix(
    tiles: LauncherTile[],
    options: { browserHostname?: string } = {},
): ReachabilityMatrixEntry[] {
    return tiles.map((tile) => {
        const declaredMode = tile.web?.mode ?? 'unavailable';
        return {
            tileId: tile.id,
            name: tile.name,
            declaredMode,
            route: tile.web?.route,
            browser: evaluateTileReachability(tile, 'browser', options),
            tauri: evaluateTileReachability(tile, 'tauri', options),
        };
    });
}
