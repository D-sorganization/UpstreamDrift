/**
 * Tests for the web launch contract resolution (issue #7461).
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
    canLaunchNativeWindow,
    isLocalHostname,
    resolveTileLaunchAction,
} from './webLaunch';
import type { WebLaunchContract } from './useLauncherManifest';

function tile(web?: WebLaunchContract) {
    return { web };
}

describe('isLocalHostname', () => {
    it('accepts loopback hostnames', () => {
        expect(isLocalHostname('localhost')).toBe(true);
        expect(isLocalHostname('127.0.0.1')).toBe(true);
        expect(isLocalHostname('[::1]')).toBe(true);
    });

    it('rejects remote hostnames', () => {
        expect(isLocalHostname('example.com')).toBe(false);
        expect(isLocalHostname('192.168.1.50')).toBe(false);
    });
});

describe('canLaunchNativeWindow', () => {
    beforeEach(() => {
        delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
    });

    it('allows when running under Tauri even on a remote host', () => {
        (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__ = {};
        expect(canLaunchNativeWindow('example.com')).toBe(true);
        delete (window as unknown as Record<string, unknown>).__TAURI_INTERNALS__;
    });

    it('allows on localhost without Tauri', () => {
        expect(canLaunchNativeWindow('localhost')).toBe(true);
    });

    it('blocks on a remote host without Tauri', () => {
        expect(canLaunchNativeWindow('example.com')).toBe(false);
    });
});

describe('resolveTileLaunchAction', () => {
    it('route mode navigates in-app', () => {
        const action = resolveTileLaunchAction(
            tile({ mode: 'route', route: '/tools/terrain' }),
            false,
        );
        expect(action).toEqual({ kind: 'navigate', route: '/tools/terrain' });
    });

    it('route mode without a route is blocked, not a dead button', () => {
        const action = resolveTileLaunchAction(tile({ mode: 'route' }), true);
        expect(action.kind).toBe('blocked');
    });

    it('native-window mode launches when allowed', () => {
        const action = resolveTileLaunchAction(
            tile({ mode: 'native-window' }),
            true,
        );
        expect(action).toEqual({ kind: 'native-window' });
    });

    it('native-window mode is blocked with Desktop app only badge when remote', () => {
        const action = resolveTileLaunchAction(
            tile({ mode: 'native-window' }),
            false,
        );
        expect(action.kind).toBe('blocked');
        if (action.kind === 'blocked') {
            expect(action.badge).toBe('Desktop app only');
            expect(action.reason).toMatch(/native desktop window/i);
        }
    });

    it('unavailable mode is blocked with the declared reason', () => {
        const action = resolveTileLaunchAction(
            tile({ mode: 'unavailable', reason: 'REST API endpoint only' }),
            true,
        );
        expect(action).toEqual({
            kind: 'blocked',
            badge: 'Unavailable',
            reason: 'REST API endpoint only',
        });
    });

    it('legacy tiles without a contract behave as native-window', () => {
        expect(resolveTileLaunchAction(tile(undefined), true)).toEqual({
            kind: 'native-window',
        });
        expect(resolveTileLaunchAction(tile(undefined), false).kind).toBe('blocked');
    });
});
