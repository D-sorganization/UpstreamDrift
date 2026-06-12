import { useState, useCallback, useRef, useEffect } from 'react';
import { getApiBase } from './backend';
import { apiFetch } from './fetch';
import { withLauncherWebSocketToken } from './websocketToken';
import type { EngineListResponse, EngineStatusResponse } from './generated/types';

/**
 * Result of a `setSpeed` call (issue #7166).
 *
 * `setSpeed` never rejects; it reports failure in-band so fire-and-forget
 * callers cannot produce an unhandled rejection.
 */
export interface SetSpeedResult {
  success: boolean;
  error?: string;
}

export interface SimulationFrame {
  frame: number;
  time: number;
  state: Record<string, number[]>;
  analysis?: {
    joint_angles?: number[];
    velocities?: number[];
  };
}

export interface SimulationConfig {
  model?: string;
  duration?: number;
  timestep?: number;
  live_analysis?: boolean;
  initial_state?: Record<string, number[]>;
}

/**
 * Engine status payload — generated from the API contract (issue #7447).
 *
 * Do NOT hand-write this shape: it mirrors `EngineStatusResponse` in
 * `src/api/models/responses.py` via `ui/src/api/generated/types.ts`.
 */
export type EngineStatus = EngineStatusResponse;

// Maximum number of frames to keep in history to prevent memory leaks
const MAX_FRAMES_HISTORY = 1000;

export type ConnectionStatus =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'failed'
  // #6896: connection dropped mid-run. The server has no resume protocol — a
  // reconnect always restarts from t=0 and wipes the timeline — so instead of
  // silently doing that we surface this state and require an explicit restart.
  | 'lost';

export async function fetchEngines(): Promise<EngineStatus[]> {
  // Compile-time contract from the generated types; the runtime shape check
  // below stays because the backend ships separately (see ui/README.md).
  let data: Partial<EngineListResponse>;
  try {
    data = await apiFetch<Partial<EngineListResponse>>('/api/engines');
  } catch {
    throw new Error('Failed to fetch engines');
  }
  if (!Array.isArray(data.engines)) {
    throw new Error('Unexpected engines response shape');
  }
  return data.engines;
}

export function useSimulation(engineType: string) {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<SimulationFrame | null>(null);
  const [frames, setFrames] = useState<SimulationFrame[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [wsError, setWsError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // Track if component is mounted to prevent state updates after unmount
  const isMountedRef = useRef(true);

  const connect = useCallback((config: SimulationConfig = {}) => {
    // Unmount guard at function entry (issue #7166): a connect() raced with
    // unmount otherwise warns about state updates on an unmounted component
    // when it reaches setConnectionStatus('connecting') below.
    if (!isMountedRef.current) return;

    // Close any existing WebSocket connection before creating a new one
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus('connecting');

    // Build WS URL: use the API base to handle Tauri vs. browser mode (issue #6637)
    const apiBase = getApiBase();
    let wsUrl: string;
    if (apiBase) {
      // Tauri: explicit backend origin, swap http(s) → ws(s)
      wsUrl = apiBase.replace(/^http/, 'ws') + `/api/ws/simulate/${engineType}`;
    } else {
      // Browser/Vite: relative URL using current page origin
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      wsUrl = `${protocol}//${host}/api/ws/simulate/${engineType}`;
    }

    const ws = new WebSocket(withLauncherWebSocketToken(wsUrl));
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMountedRef.current) return;

      setConnectionStatus('connected');
      setWsError(null);
      setIsRunning(true);
      // A fresh connection always begins a new run from t=0 (the server has no
      // resume protocol), so clearing the timeline here is correct — this path
      // is only reached on an explicit start(), never on a silent reconnect.
      setFrames([]);

      // duration/timestep come from the caller's config (the store is the
      // single source of truth, #7424) — no hardcoded simulation defaults here.
      // Only live_analysis carries a safe fallback when the caller omits it.
      ws.send(JSON.stringify({
        action: 'start',
        config: {
          live_analysis: true,
          ...config,
        },
      }));
    };

    ws.onmessage = (event) => {
      if (!isMountedRef.current) return;
      try {
        const data = JSON.parse(event.data);

        if (data.status === 'complete' || data.status === 'stopped') {
          setIsRunning(false);
          return;
        }
        if (data.status === 'paused') {
          setIsPaused(true);
          return;
        }

        if (data.frame !== undefined) {
          setCurrentFrame(data);
          // Limit frames history to prevent unbounded memory growth
          setFrames(prev => {
            const newFrames = [...prev, data];
            if (newFrames.length > MAX_FRAMES_HISTORY) {
              return newFrames.slice(-MAX_FRAMES_HISTORY);
            }
            return newFrames;
          });
        }
      } catch (err) {
        console.error("WS Parse Error", err);
      }
    };

    ws.onerror = () => {
      console.error('WebSocket error occurred');
      if (isMountedRef.current) {
        setWsError('WebSocket connection error — check server status');
      }
    };

    ws.onclose = (event) => {
      if (!isMountedRef.current) return;

      setIsRunning(false);

      // Clean close (code 1000) or user-initiated stop — nothing to recover.
      if (event.wasClean || event.code === 1000) {
        setConnectionStatus('disconnected');
        return;
      }

      // #6896: SAFE behaviour on an unclean drop.
      //
      // Previously we auto-reconnected here, and the reopened socket's onopen
      // wiped `frames` and sent {action:'start'} — which the server always runs
      // from time_elapsed=0.0 (it has no resume/offset protocol). A transient
      // blip therefore silently restarted a multi-second run from the beginning
      // while the UI implied continuity ("Reconnecting…").
      //
      // Until the server supports resume (tracked as a follow-up), we DO NOT
      // silently reconnect-and-restart. We preserve the captured frames and the
      // last frame's time, mark the connection 'lost', and require an explicit
      // user restart (calling start()) to begin a new run. This guarantees we
      // never reset frames/time without user action.
      console.warn(
        'WebSocket closed unexpectedly. Connection lost — an explicit restart ' +
          'is required (the simulation cannot resume from where it stopped).',
      );
      setConnectionStatus('lost');
      setWsError(
        'Connection lost — the simulation cannot resume. Restart to run again.',
      );
    };
  }, [engineType]);

  const start = useCallback((config: SimulationConfig = {}) => {
    // Explicit user restart: open a fresh connection. onopen clears the
    // timeline, which is the only place frames are reset (#6896).
    connect(config);
  }, [connect]);

  const stop = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'stop' }));
    }
    // Close the WebSocket connection after sending stop
    if (ws) {
      ws.close(1000, 'User stopped simulation');
      wsRef.current = null;
    }
    setConnectionStatus('disconnected');
  }, []);

  const pause = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'pause' }));
      setIsPaused(true);
    }
  }, []);

  const resume = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'resume' }));
      setIsPaused(false);
    }
  }, []);

  // Returns a structured result instead of throwing (issue #7166): callers
  // that fire-and-forget (event handlers) previously turned the re-thrown error
  // into an `unhandledrejection`. A result object makes the failure contract
  // explicit (DbC) so callers can surface it via the existing error toast.
  const setSpeed = useCallback(async (speed: number): Promise<SetSpeedResult> => {
    try {
      await apiFetch<unknown>('/api/simulation/speed', {
        method: 'POST',
        body: JSON.stringify({ speed_factor: speed }),
      });
      return { success: true };
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      return {
        success: false,
        error: `Failed to set simulation speed to ${speed}x: ${detail}`,
      };
    }
  }, []);

  // Track mounted state and cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, []);

  return {
    isRunning,
    isPaused,
    currentFrame,
    frames,
    connectionStatus,
    wsError,
    start,
    stop,
    pause,
    resume,
    setSpeed,
  };
}
