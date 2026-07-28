/**
 * Regression tests for the `apiFetch` request timeout (issue #8080).
 *
 * `window.fetch` never times out on its own. If the API accepts the TCP
 * connection but never answers, the returned promise never settles, so any
 * caller awaiting it stays in its loading state forever with no error path.
 * That is what left Motion Capture on "Loading sources..." indefinitely.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  apiFetch,
  apiFetchBlob,
  apiFetchForm,
  apiFetchRaw,
  DEFAULT_TIMEOUT_MS,
} from './fetch';

function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('apiFetch timeout (#8080)', () => {
  it('passes an AbortSignal by default so a request cannot hang forever', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetch('/api/anything');

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it('passes an AbortSignal for raw response callers', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetchRaw('/api/export');

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it('passes an AbortSignal for multipart upload callers', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ uploaded: true }));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetchForm('/api/upload', new FormData());

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(init.headers).toBeUndefined();
  });

  it('exposes a sane default timeout', () => {
    expect(DEFAULT_TIMEOUT_MS).toBeGreaterThan(0);
    expect(DEFAULT_TIMEOUT_MS).toBeLessThanOrEqual(30_000);
  });

  it('rejects with a recognisable "timed out" message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new DOMException('The operation timed out.', 'TimeoutError')),
    );

    await expect(apiFetch('/api/slow')).rejects.toThrow(/timed out/i);
  });

  it('reports the configured timeout value in the message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new DOMException('timeout', 'TimeoutError')),
    );

    await expect(apiFetch('/api/slow', { timeoutMs: 250 })).rejects.toThrow(/250ms/);
  });

  it('actually aborts a never-answering request within the timeout', async () => {
    // The signal is the only thing that can end this request.
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(
        (_url: string, init: RequestInit) =>
          new Promise((_resolve, reject) => {
            init.signal?.addEventListener('abort', () => {
              reject(new DOMException('aborted', 'TimeoutError'));
            });
          }),
      ),
    );

    await expect(apiFetch('/api/hangs', { timeoutMs: 50 })).rejects.toThrow(
      /timed out/i,
    );
  });

  it('distinguishes a caller abort from a timeout', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new DOMException('aborted', 'AbortError')),
    );

    await expect(apiFetch('/api/cancelled')).rejects.toThrow(/aborted/i);
  });

  it('allows disabling the timeout with timeoutMs: 0', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetch('/api/streaming', { timeoutMs: 0 });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeUndefined();
  });

  it('honours a caller-supplied signal alongside the timeout', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({}));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();

    await apiFetch('/api/thing', { signal: controller.signal });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
    expect(init.signal).not.toBe(controller.signal);
  });

  it('does not disturb normal success or HTTP-error handling', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ value: 7 })));
    await expect(apiFetch<{ value: number }>('/api/ok')).resolves.toEqual({ value: 7 });

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 405,
        statusText: 'Method Not Allowed',
        json: () => Promise.resolve({ detail: 'Method Not Allowed' }),
      } as unknown as Response),
    );
    await expect(apiFetch('/api/bad')).rejects.toThrow('Method Not Allowed');
  });

  it('extracts FastAPI detail for raw and blob callers', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({ detail: 'Invalid export request' }),
      } as unknown as Response),
    );

    await expect(apiFetchBlob('/api/export')).rejects.toThrow(
      'Invalid export request',
    );
  });
});

describe('apiFetchForm timeout (#8144)', () => {
  it('passes an AbortSignal by default so uploads cannot hang forever', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetchForm('/api/upload', new FormData());

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.signal).toBeInstanceOf(AbortSignal);
  });

  it('rejects with a recognisable timed-out message', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValue(new DOMException('timeout', 'TimeoutError')),
    );

    await expect(
      apiFetchForm('/api/upload', new FormData(), { timeoutMs: 250 }),
    ).rejects.toThrow(/250ms/);
  });

  it('does not set a JSON content type for multipart bodies', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ ok: true }));
    vi.stubGlobal('fetch', fetchMock);

    await apiFetchForm('/api/upload', new FormData());

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.body).toBeInstanceOf(FormData);
    expect(init.headers).toBeUndefined();
  });
});
