/**
 * useRealtimeChannel — reconnecting WebSocket subscription to a realtime
 * pub-sub channel.
 *
 * Connects to `WS /api/realtime/subscribe?channel=<name>` (see
 * `src/api/routes/realtime.py`): the server keeps the socket open and pushes
 * one JSON message per publish on the channel. This hook parses each message
 * and exposes the most recent payload.
 *
 * Reconnection: any close while mounted schedules a reconnect with
 * exponential backoff (500ms doubling, capped at 5s). A successful open
 * resets the backoff. On unmount every timer is cleared and the socket is
 * closed, so no reconnect loop can outlive the component.
 *
 * URL/auth conventions follow `useSimulation` in `ui/src/api/client.ts`:
 * `getApiBase()` handles Tauri vs. browser origins and the launcher
 * capability token (`withLauncherWebSocketToken`) satisfies the local-mode
 * WebSocket guard.
 *
 * See issue #8406.
 */

import { useEffect, useRef, useState } from 'react';
import { getApiBase } from '@/api/backend';
import {
  ensureLauncherCapabilityToken,
  withLauncherWebSocketToken,
} from '@/api/websocketToken';
import { logger } from '@/utils/logger';

/** Connection status of the realtime channel subscription. */
export type RealtimeChannelStatus =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'reconnecting';

/** Initial reconnect delay; doubles on every consecutive failure. */
const INITIAL_BACKOFF_MS = 500;

/** Upper bound on the reconnect delay (issue #8406: capped ~5s). */
const MAX_BACKOFF_MS = 5_000;

/**
 * Build the subscribe URL for a realtime channel.
 *
 * Exported for tests. Mirrors the ws/wss derivation used by the simulate
 * WebSocket in `ui/src/api/client.ts`.
 *
 * @param channel - Channel name in `scope/topic` form (e.g. `pose/canonical`).
 * @returns Absolute `ws(s)://` URL for the subscribe endpoint.
 */
export function buildRealtimeSubscribeUrl(channel: string): string {
  if (!channel || channel.trim().length === 0) {
    throw new Error('channel must be a non-empty string');
  }
  const query = `channel=${encodeURIComponent(channel)}`;
  const apiBase = getApiBase();
  if (apiBase) {
    // Tauri: explicit backend origin, swap http(s) → ws(s)
    return `${apiBase.replace(/^http/, 'ws')}/api/realtime/subscribe?${query}`;
  }
  // Browser/Vite: relative URL using current page origin
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/api/realtime/subscribe?${query}`;
}

/**
 * Subscribe to a realtime pub-sub channel and expose the latest message.
 *
 * @param channel - Channel name (`scope/topic`), e.g. `'pose/canonical'`.
 * @param enabled - When false the hook stays idle (no socket is opened).
 * @returns `message` — the latest parsed JSON payload (null until the first
 *   message arrives) — and the connection `status`.
 */
export function useRealtimeChannel<T = unknown>(
  channel: string,
  enabled: boolean = true,
): { message: T | null; status: RealtimeChannelStatus } {
  const [message, setMessage] = useState<T | null>(null);
  const [status, setStatus] = useState<RealtimeChannelStatus>('connecting');
  // The active socket, shared between connect attempts and the cleanup.
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let disposed = false;
    let retryTimer: number | null = null;
    let backoffMs = INITIAL_BACKOFF_MS;

    const scheduleReconnect = () => {
      if (disposed) return;
      setStatus('reconnecting');
      retryTimer = window.setTimeout(() => {
        retryTimer = null;
        void connect();
      }, backoffMs);
      backoffMs = Math.min(backoffMs * 2, MAX_BACKOFF_MS);
    };

    const connect = async (): Promise<void> => {
      if (disposed) return;

      // Best-effort: satisfies the local-mode WebSocket guard (#8077). A null
      // token is non-fatal — cloud deployments use bearer auth instead, and
      // the socket itself reports any real failure.
      await ensureLauncherCapabilityToken();
      if (disposed) return;
      setStatus((prev) => (prev === 'reconnecting' ? prev : 'connecting'));

      let socket: WebSocket;
      try {
        socket = new WebSocket(
          withLauncherWebSocketToken(buildRealtimeSubscribeUrl(channel)),
        );
      } catch (err) {
        logger.error('realtime: failed to open WebSocket', err);
        scheduleReconnect();
        return;
      }
      wsRef.current = socket;

      socket.onopen = () => {
        if (disposed) return;
        backoffMs = INITIAL_BACKOFF_MS;
        setStatus('connected');
      };

      socket.onmessage = (event: MessageEvent) => {
        if (disposed) return;
        try {
          setMessage(JSON.parse(event.data as string) as T);
        } catch (err) {
          // A malformed frame must not kill the subscription.
          logger.error('realtime: unparseable channel message', err);
        }
      };

      socket.onerror = () => {
        // onclose always follows onerror; reconnect is handled there.
        logger.warn(`realtime: WebSocket error on channel ${channel}`);
      };

      socket.onclose = () => {
        if (disposed) return;
        wsRef.current = null;
        // The subscribe endpoint never closes a healthy subscription, so any
        // close while mounted (including a clean server shutdown) warrants a
        // reconnect attempt.
        scheduleReconnect();
      };
    };

    void connect();

    return () => {
      disposed = true;
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
      }
      const socket = wsRef.current;
      wsRef.current = null;
      if (socket) {
        // Detach handlers first so the close cannot schedule a reconnect.
        socket.onclose = null;
        socket.onmessage = null;
        socket.onerror = null;
        try {
          socket.close(1000, 'Component unmounted');
        } catch {
          // Closing an already-dead socket must never throw during unmount.
        }
      }
    };
  }, [channel, enabled]);

  // A disabled subscription always reads as idle without needing an effect.
  return { message, status: enabled ? status : 'idle' };
}
