import { useState, useCallback, useRef, useEffect } from 'react';
import type { RunSummary } from '@/components/analysis/SummaryPanel';

export type { RunSummary };

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

export interface EngineStatus {
  name: string;
  available: boolean;
  loaded: boolean;
  version?: string;
  capabilities: string[];
}

// Maximum number of frames to keep in history to prevent memory leaks
const MAX_FRAMES_HISTORY = 1000;

// WebSocket reconnection configuration
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 1000;
const MAX_RECONNECT_DELAY_MS = 30000;

export type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'failed';

export async function fetchEngines(): Promise<EngineStatus[]> {
  try {
    const response = await fetch('/api/engines', { signal: AbortSignal.timeout(5000) });
    if (!response.ok) {
      if (response.status === 503) {
        throw new Error(
          'Backend API server is temporarily unavailable (503 Service Unavailable). ' +
          'The server may be starting up. Try again in a moment or check: python -m uvicorn src.api.main:app'
        );
      }
      throw new Error(`Failed to fetch engines: HTTP ${response.status}`);
    }
    const data = await response.json();
    return data.engines;
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      throw new Error(
        'Backend API server is not running.\n\n' +
        'To start the server:\n' +
        '  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n' +
        'Or with Docker:\n' +
        '  docker-compose up backend\n\n' +
        'Check that the server is accessible at http://localhost:8000/api/engines'
      );
    }
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error(
        'Backend API request timed out (5s). Server may be overloaded or unreachable.\n\n' +
        'Check if the server is running and responsive:\n' +
        '  curl http://localhost:8000/api/health'
      );
    }
    throw error;
  }
}

export function useSimulation(engineType: string) {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<SimulationFrame | null>(null);
  const [frames, setFrames] = useState<SimulationFrame[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [runSummary, setRunSummary] = useState<RunSummary | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // Track if component is mounted to prevent state updates after unmount
  const isMountedRef = useRef(true);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const pendingConfigRef = useRef<SimulationConfig | null>(null);

  // Calculate exponential backoff delay
  const getReconnectDelay = useCallback((attempt: number): number => {
    const delay = Math.min(
      BASE_RECONNECT_DELAY_MS * Math.pow(2, attempt),
      MAX_RECONNECT_DELAY_MS
    );
    // Add jitter to prevent thundering herd
    return delay + Math.random() * 1000;
  }, []);

  // Clear any pending reconnect timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  // Ref to store connect function for reconnection logic (avoids circular reference)
  const connectRef = useRef<((config: SimulationConfig) => void) | null>(null);


  const connect = useCallback((config: SimulationConfig = {}) => {
    // Store config for potential reconnection
    pendingConfigRef.current = config;

    // Close any existing WebSocket connection before creating a new one
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setConnectionStatus('connecting');

    // Determine WS protocol based on current connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/ws/simulate/${engineType}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMountedRef.current) return;

      // Reset reconnection attempts on successful connection
      reconnectAttemptsRef.current = 0;
      setConnectionStatus('connected');
      setIsRunning(true);
      setFrames([]);
      setRunSummary(null);

      ws.send(JSON.stringify({
        action: 'start',
        config: {
          duration: 3.0,
          timestep: 0.002,
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
          if (data.status === 'complete' && data.summary != null) {
            setRunSummary(data.summary as RunSummary);
          }
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

    ws.onerror = (error) => {
      const errorMsg = error instanceof Event ? 'WebSocket connection error' : String(error);
      if (!isMountedRef.current) return;

      // Determine if this is a connection refused error
      const isConnectionRefused =
        wsUrl.includes('localhost') ||
        wsUrl.includes('127.0.0.1');

      const detailedMsg = isConnectionRefused
        ? `WebSocket error: Failed to connect to backend at ${wsUrl}.\n\n` +
          `The backend API server may not be running.\n` +
          `Start it with:\n` +
          `  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n` +
          `Check that the server is reachable at http://localhost:8000/api/health`
        : `WebSocket error: ${errorMsg}\n` +
          `Cannot establish connection to ${wsUrl}.\n` +
          `Check that the backend server is running and accessible.`;

      console.error(detailedMsg);
    };

    ws.onclose = (event) => {
      if (!isMountedRef.current) return;

      setIsRunning(false);

      // Don't reconnect if this was a clean close (code 1000) or user-initiated
      if (event.wasClean || event.code === 1000) {
        setConnectionStatus('disconnected');
        reconnectAttemptsRef.current = 0;
        return;
      }

      // Attempt reconnection with exponential backoff
      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = getReconnectDelay(reconnectAttemptsRef.current);
        console.log(`WebSocket closed unexpectedly. Reconnecting in ${Math.round(delay)}ms (attempt ${reconnectAttemptsRef.current + 1}/${MAX_RECONNECT_ATTEMPTS})`);

        setConnectionStatus('reconnecting');

        reconnectTimeoutRef.current = setTimeout(() => {
          if (isMountedRef.current && connectRef.current) {
            reconnectAttemptsRef.current++;
            connectRef.current(pendingConfigRef.current || {});
          }
        }, delay);
      } else {
        console.error('Max reconnection attempts reached. Connection failed.');
        setConnectionStatus('failed');
        reconnectAttemptsRef.current = 0;
      }
    };
  }, [engineType, getReconnectDelay]);

  // Sync connectRef with latest connect function (in effect, not during render)
  useEffect(() => {
    connectRef.current = connect;
  }, [connect]);

  const start = useCallback((config: SimulationConfig = {}) => {
    // Reset reconnection state when starting fresh
    clearReconnectTimeout();
    reconnectAttemptsRef.current = 0;
    connect(config);
  }, [connect, clearReconnectTimeout]);

  const stop = useCallback(() => {
    // Clear any pending reconnection
    clearReconnectTimeout();
    reconnectAttemptsRef.current = 0;

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
  }, [clearReconnectTimeout]);

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

  // Track mounted state and cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      clearReconnectTimeout();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
        wsRef.current = null;
      }
    };
  }, [clearReconnectTimeout]);

  return {
    isRunning,
    isPaused,
    currentFrame,
    frames,
    connectionStatus,
    runSummary,
    start,
    stop,
    pause,
    resume,
  };
}
