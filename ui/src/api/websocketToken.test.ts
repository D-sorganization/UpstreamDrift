/**
 * Regression tests for launcher capability token acquisition (issue #8077).
 *
 * The backend guard `enforce_local_websocket_guard` rejects any local-mode
 * WebSocket upgrade that does not carry the token issued by
 * `GET /api/launcher/manifest`. The token used to be cached only as a side
 * effect of the Dashboard's `useLauncherManifest` hook, so a user who opened
 * `/simulation` directly got an instant
 * "Connection lost — restart required" on Start.
 *
 * Verified against a live server before writing these:
 *   ws://localhost:8010/api/ws/simulate/pendulum            -> HTTP 403
 *   ws://localhost:8010/...?launcher_token=<manifest token> -> HTTP 101
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  ensureLauncherCapabilityToken,
  getLauncherCapabilityToken,
  setLauncherCapabilityToken,
  withLauncherWebSocketToken,
} from './websocketToken';

function manifestResponse(body: unknown, ok = true): Response {
  return {
    ok,
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

beforeEach(() => {
  setLauncherCapabilityToken(null);
});

afterEach(() => {
  setLauncherCapabilityToken(null);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('ensureLauncherCapabilityToken (#8077)', () => {
  it('fetches the manifest and caches the token when none is known', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(manifestResponse({ launcher_csrf_token: 'tok-abc' }));
    vi.stubGlobal('fetch', fetchMock);

    const token = await ensureLauncherCapabilityToken();

    expect(token).toBe('tok-abc');
    expect(getLauncherCapabilityToken()).toBe('tok-abc');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/launcher/manifest');
  });

  it('does not re-fetch once the token is cached', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(manifestResponse({ launcher_csrf_token: 'tok-abc' }));
    vi.stubGlobal('fetch', fetchMock);

    await ensureLauncherCapabilityToken();
    await ensureLauncherCapabilityToken();
    await ensureLauncherCapabilityToken();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('shares one in-flight request between concurrent callers', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(manifestResponse({ launcher_csrf_token: 'tok-abc' }));
    vi.stubGlobal('fetch', fetchMock);

    const results = await Promise.all([
      ensureLauncherCapabilityToken(),
      ensureLauncherCapabilityToken(),
      ensureLauncherCapabilityToken(),
    ]);

    expect(results).toEqual(['tok-abc', 'tok-abc', 'tok-abc']);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('is a no-op when a token was already supplied by the manifest hook', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    setLauncherCapabilityToken('preset-token');

    await expect(ensureLauncherCapabilityToken()).resolves.toBe('preset-token');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('resolves to null instead of throwing on a network error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('ECONNREFUSED')));

    await expect(ensureLauncherCapabilityToken()).resolves.toBeNull();
    expect(getLauncherCapabilityToken()).toBeNull();
  });

  it('resolves to null on a non-2xx manifest response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(manifestResponse({}, false)));

    await expect(ensureLauncherCapabilityToken()).resolves.toBeNull();
  });

  it('resolves to null when the manifest omits the token', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(manifestResponse({ tiles: [], version: '1' })),
    );

    await expect(ensureLauncherCapabilityToken()).resolves.toBeNull();
  });

  it('resolves to null when the token is present but empty', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(manifestResponse({ launcher_csrf_token: '' })),
    );

    await expect(ensureLauncherCapabilityToken()).resolves.toBeNull();
  });

  it('retries after a failure rather than caching the failure', async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValueOnce(manifestResponse({ launcher_csrf_token: 'later' }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(ensureLauncherCapabilityToken()).resolves.toBeNull();
    await expect(ensureLauncherCapabilityToken()).resolves.toBe('later');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('applies a timeout so a hung manifest cannot block a connect forever', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(manifestResponse({ launcher_csrf_token: 'tok' }));
    vi.stubGlobal('fetch', fetchMock);

    await ensureLauncherCapabilityToken();

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeDefined();
  });
});

describe('withLauncherWebSocketToken', () => {
  it('appends the cached token as launcher_token', () => {
    setLauncherCapabilityToken('tok-xyz');
    const url = withLauncherWebSocketToken('ws://localhost:5180/api/ws/simulate/mujoco');
    expect(new URL(url).searchParams.get('launcher_token')).toBe('tok-xyz');
  });

  it('leaves the URL untouched when no token is cached', () => {
    const raw = 'ws://localhost:5180/api/ws/simulate/mujoco';
    expect(withLauncherWebSocketToken(raw)).toBe(raw);
  });

  it('rejects an empty URL', () => {
    expect(() => withLauncherWebSocketToken('')).toThrow(/non-empty/);
  });
});
