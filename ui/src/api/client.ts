import { useState, useCallback, useRef, useEffect } from 'react';

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
  const response = await fetch('/api/engines');
  if (!response.ok) {
    throw new Error('Failed to fetch engines');
  }
  const data = await response.json();
  return data.engines;
}

export function useSimulation(engineType: string) {
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<SimulationFrame | null>(null);
  const [frames, setFrames] = useState<SimulationFrame[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
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

  // Use a ref to hold the connect function to handle recursion
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
      console.error('WebSocket error:', error);
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
          if (isMountedRef.current) {
            reconnectAttemptsRef.current++;
            // Use ref to call connect recursively
            connectRef.current?.(pendingConfigRef.current || {});
          }
        }, delay);
      } else {
        console.error('Max reconnection attempts reached. Connection failed.');
        setConnectionStatus('failed');
        reconnectAttemptsRef.current = 0;
      }
    };
  }, [engineType, getReconnectDelay]);

  // Update the ref whenever connect changes
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
    start,
    stop,
    pause,
    resume
  };
}
