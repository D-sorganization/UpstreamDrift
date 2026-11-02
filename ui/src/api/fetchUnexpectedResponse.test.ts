/**
 * Non-JSON API responses become a typed `ApiError` (issue #14256).
 *
 * When a request falls through to the SPA (404 or a dev-proxy miss), the
 * server answers with `index.html`. `response.json()` then threw
 * `Unexpected token '<', "<!doctype "... is not valid JSON`, and pages showed
 * that parser text verbatim.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  apiFetch,
  apiFetchForm,
  describeApiError,
} from './fetch';

const HTML = '<!doctype html><html><body><div id="root"></div></body></html>';

function htmlResponse(status = 200): Response {
  return new Response(HTML, {
    status,
    headers: { 'content-type': 'text/html; charset=utf-8' },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('apiFetch on a non-JSON response (#14256)', () => {
  it('throws ApiError(kind unexpected-response) instead of a SyntaxError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(htmlResponse()));

    const err = await apiFetch('/tools/golf-simulator/destinations').catch(
      (e: unknown) => e,
    );

    expect(err).toBeInstanceOf(ApiError);
    expect(err).not.toBeInstanceOf(SyntaxError);
    const apiErr = err as ApiError;
    expect(apiErr.kind).toBe('unexpected-response');
    expect(apiErr.status).toBe(200);
    expect(apiErr.url).toContain('/tools/golf-simulator/destinations');
    expect(apiErr.message).not.toMatch(/Unexpected token|not valid JSON/);
    expect(apiErr.message).toMatch(/text\/html/);
  });

  it('treats an unparseable body without a content-type the same way', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers(),
        json: () => Promise.reject(new SyntaxError("Unexpected token '<'")),
      } as unknown as Response),
    );

    await expect(apiFetch('/api/thing')).rejects.toMatchObject({
      name: 'ApiError',
      kind: 'unexpected-response',
    });
  });

  it('still parses JSON bodies, including +json media types', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('{"ok":true}', {
          status: 200,
          headers: { 'content-type': 'application/problem+json' },
        }),
      ),
    );

    await expect(apiFetch('/api/thing')).resolves.toEqual({ ok: true });
  });

  it('applies to form uploads as well', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(htmlResponse()));

    await expect(apiFetchForm('/api/upload', new FormData())).rejects.toBeInstanceOf(
      ApiError,
    );
  });
});

describe('describeApiError (#14256)', () => {
  it('maps an unexpected response to plain-language copy for the service', () => {
    const err = new ApiError('unexpected-response', 200, '/x', 'detail');
    expect(describeApiError(err, 'Simulator', 'Failed')).toBe(
      'Simulator API not reachable',
    );
  });

  it('keeps other error messages and falls back for non-errors', () => {
    expect(describeApiError(new Error('Shot rejected'), 'Simulator', 'Failed')).toBe(
      'Shot rejected',
    );
    expect(describeApiError('boom', 'Simulator', 'Failed')).toBe('Failed');
  });
});
