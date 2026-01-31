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
  const wsRef = useRef<WebSocket | null>(null);
  // Track if component is mounted to prevent state updates after unmount
  const isMountedRef = useRef(true);

  const start = useCallback((config: SimulationConfig = {}) => {
    // Close any existing WebSocket connection before creating a new one
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Determine WS protocol based on current connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // e.g. localhost:3000 or localhost:8000
    // If running in dev via proxy, this works. If built static, works if same origin.
    // We'll trust the proxy config or relative path.
    const wsUrl = `${protocol}//${host}/api/ws/simulate/${engineType}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!isMountedRef.current) return;
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
      if (isMountedRef.current) {
        setIsRunning(false);
      }
    };

    ws.onclose = () => {
      if (isMountedRef.current) {
        setIsRunning(false);
      }
    };
  }, [engineType]);

  const stop = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'stop' }));
    }
    // Close the WebSocket connection after sending stop
    if (ws) {
      ws.close();
      wsRef.current = null;
    }
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

  // Track mounted state and cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  return {
    isRunning,
    isPaused,
    currentFrame,
    frames,
    start,
    stop,
    pause,
    resume
  };
}
