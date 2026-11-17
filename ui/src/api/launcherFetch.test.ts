import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { apiFetch, apiFetchBlob, apiFetchForm } from './fetch';
import { LAUNCHER_CAPABILITY_HEADER, launcherFetch, withLauncherCapability } from './launcherFetch';
import { getLauncherCapabilityToken, setLauncherCapabilityToken } from './websocketToken';

const URL_UNDER_TEST = 'http://127.0.0.1:8000/api/engines';

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function sentHeaders(call: unknown[]): Record<string, string> {
  const init = call[1] as RequestInit | undefined;
  return (init?.headers ?? {}) as Record<string, string>;
}

describe('launcherFetch', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
    setLauncherCapabilityToken(null);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setLauncherCapabilityToken(null);
  });

  it('sends the cached capability header', async () => {
    setLauncherCapabilityToken('cap-1');
    fetchMock.mockResolvedValueOnce(jsonResponse(200, {}));

    await launcherFetch(URL_UNDER_TEST, { method: 'POST' });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'POST' });
    expect(sentHeaders(fetchMock.mock.calls[0])[LAUNCHER_CAPABILITY_HEADER]).toBe('cap-1');
  });

  it('keeps the caller headers next to the capability header', async () => {
    setLauncherCapabilityToken('cap-1');
    fetchMock.mockResolvedValueOnce(jsonResponse(200, {}));

    await launcherFetch(URL_UNDER_TEST, { headers: { 'Content-Type': 'application/json' } });

    expect(sentHeaders(fetchMock.mock.calls[0])).toEqual({
      'Content-Type': 'application/json',
      [LAUNCHER_CAPABILITY_HEADER]: 'cap-1',
    });
  });

  it('makes one plain request when no capability is cached', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, {}));

    await launcherFetch(URL_UNDER_TEST);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledWith(URL_UNDER_TEST);
  });

  it('reads a fresh capability and retries once after a 403', async () => {
    setLauncherCapabilityToken('stale');
    fetchMock
      .mockResolvedValueOnce(jsonResponse(403, { detail: 'refused' }))
      .mockResolvedValueOnce(jsonResponse(200, { launcher_csrf_token: 'fresh' }))
      .mockResolvedValueOnce(jsonResponse(200, { ok: true }));

    const response = await launcherFetch(URL_UNDER_TEST);

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(String(fetchMock.mock.calls[1][0])).toContain('/api/launcher/manifest');
    expect(sentHeaders(fetchMock.mock.calls[2])[LAUNCHER_CAPABILITY_HEADER]).toBe('fresh');
    expect(getLauncherCapabilityToken()).toBe('fresh');
  });

  it('returns the 403 when the backend issues no newer capability', async () => {
    setLauncherCapabilityToken('same');
    fetchMock
      .mockResolvedValueOnce(jsonResponse(403, { detail: 'refused' }))
      .mockResolvedValueOnce(jsonResponse(200, { launcher_csrf_token: 'same' }));

    const response = await launcherFetch(URL_UNDER_TEST);

    expect(response.status).toBe(403);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('returns the 403 when the manifest cannot be read', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(403, { detail: 'refused' }))
      .mockRejectedValueOnce(new Error('offline'));

    const response = await launcherFetch(URL_UNDER_TEST);

    expect(response.status).toBe(403);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe('withLauncherCapability', () => {
  it('returns the same options when there is no capability', () => {
    const init = { method: 'GET' };
    expect(withLauncherCapability(init, null)).toBe(init);
    expect(withLauncherCapability(undefined, null)).toBeUndefined();
  });

  it('accepts a Headers instance', () => {
    const init = withLauncherCapability({ headers: new Headers({ Accept: 'text/plain' }) }, 'cap');
    expect(init?.headers).toEqual({ accept: 'text/plain', [LAUNCHER_CAPABILITY_HEADER]: 'cap' });
  });
});

describe('API helpers send the capability header', () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal('fetch', fetchMock);
    setLauncherCapabilityToken('cap-1');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setLauncherCapabilityToken(null);
  });

  it('apiFetch', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ok: true }));
    await apiFetch('/api/engines');
    expect(sentHeaders(fetchMock.mock.calls[0])[LAUNCHER_CAPABILITY_HEADER]).toBe('cap-1');
  });

  it('apiFetchForm', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ok: true }));
    await apiFetchForm('/api/upload', new FormData());
    expect(sentHeaders(fetchMock.mock.calls[0])[LAUNCHER_CAPABILITY_HEADER]).toBe('cap-1');
  });

  it('apiFetchBlob', async () => {
    fetchMock.mockResolvedValueOnce(new Response('data', { status: 200 }));
    await apiFetchBlob('/api/export');
    expect(sentHeaders(fetchMock.mock.calls[0])[LAUNCHER_CAPABILITY_HEADER]).toBe('cap-1');
  });
});
