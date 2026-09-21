/**
 * Acceptance Suite for Launcher Reachability (ORG-04, Issue #10513).
 *
 * Tests the reachability contract:
 * 1. RED: /tools/movement-optimizer is rejected when no actual page exists.
 * 2. RED: route/native/unavailable cases work for browser and Tauri without silently falling through to 404.
 * 3. GREEN: each advertised route mounts and its primary action is wired or explicitly blocked by a real dependency;
 *    test failures and duplicate-window prevention.
 */

import { Suspense } from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { RoutedContent } from '../App';
import { KNOWN_APP_ROUTES, isKnownAppRoute } from '../routes';
import { resolveTileLaunchAction } from './webLaunch';
import {
    evaluateTileReachability,
    generateReachabilityMatrix,
} from './launcherReachability';
import {
    loadLauncherWindowRecords,
    recordLauncherWindowLaunch,
    LAUNCHER_WINDOW_REGISTRY_KEY,
} from './launcherWindowRegistry';
import type { LauncherTile } from './useLauncherManifest';

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

const mockTile = (
    id: string,
    name: string,
    web: LauncherTile['web'],
): LauncherTile => ({
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
    web,
});

describe('ORG-04: Launcher Reachability Acceptance Tests', () => {
    beforeEach(() => {
        installMemoryStorage();
        window.localStorage.removeItem(LAUNCHER_WINDOW_REGISTRY_KEY);
        delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
    });

    // =========================================================================
    // RED Acceptance Case 1: /tools/movement-optimizer rejected when no page exists
    // =========================================================================
    describe('RED Case 1: Movement Optimizer Route Rejection', () => {
        it('rejects /tools/movement-optimizer as an invalid route', () => {
            expect(isKnownAppRoute('/tools/movement-optimizer')).toBe(false);
            expect(KNOWN_APP_ROUTES).not.toContain('/tools/movement-optimizer');
        });

        it('blocks launch action when a tile declares /tools/movement-optimizer as a web route', () => {
            const badTile = mockTile('tools_movement_optimizer', 'Movement Optimizer', {
                mode: 'route',
                route: '/tools/movement-optimizer',
            });

            const action = resolveTileLaunchAction(badTile, true);
            expect(action.kind).toBe('blocked');
            if (action.kind === 'blocked') {
                expect(action.badge).toBe('Unavailable');
                expect(action.reason).toMatch(/no web route exists/i);
            }
        });

        it('evaluates reachability as unavailable for unrouted /tools/movement-optimizer', () => {
            const badTile = mockTile('tools_movement_optimizer', 'Movement Optimizer', {
                mode: 'route',
                route: '/tools/movement-optimizer',
            });

            const browserEval = evaluateTileReachability(badTile, 'browser');
            expect(browserEval.isReachable).toBe(false);
            expect(browserEval.effectiveDestination).toBe('unavailable');

            const tauriEval = evaluateTileReachability(badTile, 'tauri');
            expect(tauriEval.isReachable).toBe(false);
            expect(tauriEval.effectiveDestination).toBe('unavailable');
        });

        it('navigating router to /tools/movement-optimizer renders 404 NotFoundPage', async () => {
            render(
                <MemoryRouter initialEntries={['/tools/movement-optimizer']}>
                    <Suspense fallback={<div>Loading...</div>}>
                        <RoutedContent />
                    </Suspense>
                </MemoryRouter>,
            );

            await waitFor(() => {
                expect(screen.getByText('Page not found')).toBeInTheDocument();
            });
            expect(screen.getByText('404')).toBeInTheDocument();
        });
    });

    // =========================================================================
    // RED Acceptance Case 2: Route, Native, Unavailable destinations work honestly
    // =========================================================================
    describe('RED Case 2: Route, Native-Window, and Unavailable Destinations', () => {
        it('route mode with valid route navigates in both browser and Tauri without 404', () => {
            const validTile = mockTile('model_explorer', 'Model Explorer', {
                mode: 'route',
                route: '/tools/model-explorer',
            });

            const browserAction = resolveTileLaunchAction(validTile, false);
            expect(browserAction).toEqual({
                kind: 'navigate',
                route: '/tools/model-explorer',
            });

            const tauriAction = resolveTileLaunchAction(validTile, true);
            expect(tauriAction).toEqual({
                kind: 'navigate',
                route: '/tools/model-explorer',
            });
        });

        it('native-window mode is launchable in Tauri and blocked with honest reason in remote browser', () => {
            const nativeTile = mockTile('movement_optimizer', 'Movement Optimizer', {
                mode: 'native-window',
            });

            // Remote browser context (nativeWindowAllowed = false)
            const browserAction = resolveTileLaunchAction(nativeTile, false);
            expect(browserAction.kind).toBe('blocked');
            if (browserAction.kind === 'blocked') {
                expect(browserAction.badge).toBe('Desktop app only');
                expect(browserAction.reason).toMatch(/desktop app|locally/i);
            }

            // Tauri context (nativeWindowAllowed = true)
            const tauriAction = resolveTileLaunchAction(nativeTile, true);
            expect(tauriAction).toEqual({ kind: 'native-window' });
        });

        it('unavailable mode provides honest explanation in both browser and Tauri', () => {
            const unavailableTile = mockTile('swing_optimizer', 'Swing Optimizer', {
                mode: 'unavailable',
                reason: 'Library-only algorithm; interactive GUI planned under ORG-16',
            });

            const browserEval = evaluateTileReachability(unavailableTile, 'browser');
            expect(browserEval.isReachable).toBe(false);
            expect(browserEval.reason).toBe(
                'Library-only algorithm; interactive GUI planned under ORG-16',
            );

            const tauriEval = evaluateTileReachability(unavailableTile, 'tauri');
            expect(tauriEval.isReachable).toBe(false);
            expect(tauriEval.reason).toBe(
                'Library-only algorithm; interactive GUI planned under ORG-16',
            );
        });
    });

    // =========================================================================
    // GREEN Acceptance Case 3: Reachability Matrix & Duplicate-Window Prevention
    // =========================================================================
    describe('GREEN Case 3: Reachability Matrix & Window Lifecycle', () => {
        it('generates reachability matrix covering every effective tile ID and both contexts', () => {
            const tiles: LauncherTile[] = [
                mockTile('model_explorer', 'Model Explorer', {
                    mode: 'route',
                    route: '/tools/model-explorer',
                }),
                mockTile('movement_optimizer', 'Movement Optimizer', {
                    mode: 'native-window',
                }),
                mockTile('injury_analysis', 'Injury Analysis', {
                    mode: 'unavailable',
                    reason: 'Library only',
                }),
            ];

            const matrix = generateReachabilityMatrix(tiles, {
                browserHostname: 'cloud.upstreamdrift.org',
            });

            expect(matrix).toHaveLength(3);

            // Tile 1: Route
            expect(matrix[0].tileId).toBe('model_explorer');
            expect(matrix[0].browser.isReachable).toBe(true);
            expect(matrix[0].tauri.isReachable).toBe(true);

            // Tile 2: Native Window
            expect(matrix[1].tileId).toBe('movement_optimizer');
            expect(matrix[1].browser.isReachable).toBe(false);
            expect(matrix[1].browser.reason).toMatch(/desktop app/i);
            expect(matrix[1].tauri.isReachable).toBe(true);

            // Tile 3: Unavailable
            expect(matrix[2].tileId).toBe('injury_analysis');
            expect(matrix[2].browser.isReachable).toBe(false);
            expect(matrix[2].tauri.isReachable).toBe(false);
            expect(matrix[2].browser.reason).toBe('Library only');
        });

        it('prevents duplicate windows and preserves focus/reuse behavior for launched tiles', () => {
            const tile = mockTile('native_tool', 'Native Tool', {
                mode: 'native-window',
            });

            // First launch
            const t1 = new Date('2026-09-19T10:00:00Z');
            const records1 = recordLauncherWindowLaunch(tile, { now: t1 });
            expect(records1).toHaveLength(1);
            expect(records1[0].launchCount).toBe(1);
            expect(records1[0].launchedAt).toBe('2026-09-19T10:00:00.000Z');
            expect(records1[0].focusedAt).toBe('2026-09-19T10:00:00.000Z');

            // Second launch (re-focusing existing window)
            const t2 = new Date('2026-09-19T10:05:00Z');
            const records2 = recordLauncherWindowLaunch(tile, { now: t2 });
            expect(records2).toHaveLength(1); // No duplicate window record created!
            expect(records2[0].launchCount).toBe(2);
            expect(records2[0].launchedAt).toBe('2026-09-19T10:00:00.000Z'); // preserved
            expect(records2[0].focusedAt).toBe('2026-09-19T10:05:00.000Z'); // updated focus

            const stored = loadLauncherWindowRecords();
            expect(stored).toHaveLength(1);
            expect(stored[0].launchCount).toBe(2);
        });
    });
});
