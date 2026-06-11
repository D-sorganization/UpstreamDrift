const LAUNCHER_WS_TOKEN_QUERY = 'launcher_token';

let launcherCapabilityToken: string | null = null;

/**
 * Cache the launcher capability token for local WebSocket handshakes.
 *
 * The token is issued by `/api/launcher/manifest` and is only meaningful for
 * same-session local launcher traffic.
 */
export function setLauncherCapabilityToken(token: string | null | undefined): void {
  launcherCapabilityToken = token && token.length > 0 ? token : null;
}

export function getLauncherCapabilityToken(): string | null {
  return launcherCapabilityToken;
}

export function withLauncherWebSocketToken(url: string): string {
  if (!url || url.trim().length === 0) {
    throw new Error('url must be a non-empty string');
  }
  const token = getLauncherCapabilityToken();
  if (!token) {
    return url;
  }
  const parsed = new URL(url, window.location.href);
  parsed.searchParams.set(LAUNCHER_WS_TOKEN_QUERY, token);
  return parsed.toString();
}
