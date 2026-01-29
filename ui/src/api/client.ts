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

  const start = useCallback((config: SimulationConfig = {}) => {
    // Determine WS protocol based on current connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // e.g. localhost:3000 or localhost:8000
    // If running in dev via proxy, this works. If built static, works if same origin.
    // We'll trust the proxy config or relative path.
    const wsUrl = `${protocol}//${host}/api/ws/simulate/${engineType}`;
    
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
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
            setFrames(prev => [...prev, data]);
          }
      } catch (err) {
          console.error("WS Parse Error", err);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsRunning(false);
    };

    ws.onclose = () => {
      setIsRunning(false);
    };
  }, [engineType]);

  const stop = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'stop' }));
  }, []);

  const pause = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'pause' }));
    setIsPaused(true);
  }, []);

  const resume = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'resume' }));
    setIsPaused(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
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
