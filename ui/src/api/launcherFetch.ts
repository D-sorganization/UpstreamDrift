/**
 * `fetch` for backend API requests that attaches the launcher capability.
 *
 * The local desktop backend issues a capability token through
 * `GET /api/launcher/manifest`. Requests carry it in the
 * `X-Launcher-CSRF-Token` header. Hosted deployments issue no token; there the
 * header is simply absent and bearer authentication applies instead.
 *
 * The token is normally cached at application start. If it is missing or the
 * backend was restarted (which issues a new token), the backend answers 403;
 * the request is then retried once with a freshly read token.
 */

import {
  getLauncherCapabilityToken,
  refreshLauncherCapabilityToken,
} from './websocketToken';

export const LAUNCHER_CAPABILITY_HEADER = 'X-Launcher-CSRF-Token';

const HTTP_FORBIDDEN = 403;

function headersAsRecord(headers: HeadersInit | undefined): Record<string, string> {
  if (!headers) return {};
  if (headers instanceof Headers || Array.isArray(headers)) {
    return Object.fromEntries(new Headers(headers));
  }
  return { ...headers };
}

/**
 * Return `init` with the capability header added.
 *
 * @postcondition With no token, `init` is returned unchanged (same reference).
 */
export function withLauncherCapability(
  init: RequestInit | undefined,
  token: string | null,
): RequestInit | undefined {
  if (!token) return init;
  return {
    ...init,
    headers: { ...headersAsRecord(init?.headers), [LAUNCHER_CAPABILITY_HEADER]: token },
  };
}

function send(input: RequestInfo | URL, init: RequestInit | undefined): Promise<Response> {
  // Keep the one-argument call shape when there are no options.
  return init === undefined ? fetch(input) : fetch(input, init);
}

/**
 * `fetch` with the launcher capability header.
 *
 * @param input - Request URL
 * @param init - Optional request options; a streaming `body` cannot be retried
 * @returns The response. A 403 is returned as-is when no newer token exists.
 */
export async function launcherFetch(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> {
  const sentToken = getLauncherCapabilityToken();
  const response = await send(input, withLauncherCapability(init, sentToken));
  if (response.status !== HTTP_FORBIDDEN) {
    return response;
  }
  const freshToken = await refreshLauncherCapabilityToken();
  if (!freshToken || freshToken === sentToken) {
    return response;
  }
  return send(input, withLauncherCapability(init, freshToken));
}
