/**
 * useChat — WebSocket hook for AI assistant chat streaming.
 *
 * Connects to /api/ws/chat/{sessionId}, handles chunk streaming with
 * exponential-backoff reconnection, and syncs into useChatStore.
 *
 * Protocol (server → client):
 *   {"type": "session_info", "session_id": "..."}
 *   {"type": "chunk", "content": "..."}
 *   {"type": "complete", "session_id": "..."}
 *   {"type": "error", "detail": "..."}
 *
 * Protocol (client → server):
 *   {"action": "send", "message": "...", "engine_context": "..."}
 *   {"action": "history"}
 *   {"action": "new_session"}
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { useChatStore } from '../stores/useChatStore';

// ── Constants ─────────────────────────────────────────────────────────────

const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30_000;

// ── Types ─────────────────────────────────────────────────────────────────

export type ChatConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'failed'
  | 'no_provider';

interface ServerMessage {
  type: 'session_info' | 'chunk' | 'complete' | 'error' | 'history' | 'session_created';
  session_id?: string;
  content?: string;
  detail?: string;
  messages?: unknown[];
}

// ── Hook ──────────────────────────────────────────────────────────────────

export function useChat() {
  const [connectionStatus, setConnectionStatus] = useState<ChatConnectionStatus>('disconnected');

  const { appendChunk, finaliseStream, addMessage, setSessionId, sessionId } = useChatStore();

  const wsRef = useRef<WebSocket | null>(null);
  const isMountedRef = useRef(true);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const connectRef = useRef<(() => void) | null>(null);

  const getReconnectDelay = useCallback((attempt: number): number => {
    const delay = Math.min(
      BASE_RECONNECT_DELAY_MS * Math.pow(2, attempt),
      MAX_RECONNECT_DELAY_MS,
    );
    return delay + Math.random() * 1000;
  }, []);

  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current !== null) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus('connecting');

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/ws/chat/new`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMountedRef.current) return;
      reconnectAttemptsRef.current = 0;
      setConnectionStatus('connected');
    };

    ws.onmessage = (event: MessageEvent) => {
      if (!isMountedRef.current) return;
      let data: ServerMessage;
      try {
        data = JSON.parse(event.data as string) as ServerMessage;
      } catch {
        return;
      }

      if (data.type === 'session_info' && data.session_id) {
        setSessionId(data.session_id);
        return;
      }

      if (data.type === 'chunk' && data.content !== undefined) {
        appendChunk(data.content);
        return;
      }

      if (data.type === 'complete') {
        finaliseStream();
        return;
      }

      if (data.type === 'error') {
        finaliseStream();
        const detail = data.detail ?? 'Unknown error';
        // Detect unconfigured provider state
        const detailLower = detail.toLowerCase();
        if (
          detailLower.includes('no provider') ||
          detailLower.includes('api key') ||
          detailLower.includes('configure') ||
          detailLower.includes('no api')
        ) {
          setConnectionStatus('no_provider');
        } else {
          addMessage('assistant', `Error: ${detail}`);
        }
        return;
      }
    };

    ws.onerror = () => {
      // onclose will handle reconnection
    };

    ws.onclose = (event: CloseEvent) => {
      if (!isMountedRef.current) return;

      if (event.wasClean || event.code === 1000) {
        setConnectionStatus('disconnected');
        reconnectAttemptsRef.current = 0;
        return;
      }

      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = getReconnectDelay(reconnectAttemptsRef.current);
        setConnectionStatus('reconnecting');

        reconnectTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current && connectRef.current) {
            reconnectAttemptsRef.current++;
            connectRef.current();
          }
        }, delay);
      } else {
        setConnectionStatus('failed');
        reconnectAttemptsRef.current = 0;
      }
    };
  }, [appendChunk, finaliseStream, addMessage, setSessionId, getReconnectDelay]);

  // Keep connectRef current
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  // Auto-connect on mount
  useEffect(() => {
    isMountedRef.current = true;
    connect();
    return () => {
      isMountedRef.current = false;
      clearReconnectTimeout();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
    // connect is stable across renders; eslint-disable-next-line is intentional
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendMessage = useCallback(
    (content: string): boolean => {
      const trimmed = content.trim();
      if (!trimmed) return false;

      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return false;

      addMessage('user', trimmed);
      ws.send(
        JSON.stringify({
          action: 'send',
          message: trimmed,
        }),
      );
      return true;
    },
    [addMessage],
  );

  const disconnect = useCallback(() => {
    clearReconnectTimeout();
    reconnectAttemptsRef.current = 0;
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }
    setConnectionStatus('disconnected');
  }, [clearReconnectTimeout]);

  return { connectionStatus, sendMessage, disconnect, sessionId };
}
