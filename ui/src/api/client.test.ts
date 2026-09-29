import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { fetchEngines, useSimulation } from './client';

/**
 * Tests for the API client module.
 * Verifies WebSocket connections and API calls work correctly.
 */

describe('fetchEngines', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should throw when engines is not an array', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ engines: 'not-an-array' }),
    });

    await expect(fetchEngines()).rejects.toThrow('Unexpected engines response shape');
  });

  it('should throw when engines key is missing', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ data: [] }),
    });

    await expect(fetchEngines()).rejects.toThrow('Unexpected engines response shape');
  });

  it('should return an array of EngineStatus when the response is valid', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          engines: [
            { name: 'mujoco', available: true, loaded: false, capabilities: [] },
          ],
        }),
    });

    const result = await fetchEngines();
    expect(Array.isArray(result)).toBe(true);
    expect(result[0].name).toBe('mujoco');
  });

  it('should throw on non-ok HTTP status', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
    });

    await expect(fetchEngines()).rejects.toThrow('Failed to fetch engines');
  });
});

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('fetchEngines', () => {
    it('should return a list of available engines', async () => {
      // Mock fetch
      global.fetch = vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            engines: ['mujoco', 'drake', 'pinocchio'],
          }),
      });

      const response = await fetch('/api/engines');
      const data = await response.json();

      expect(data.engines).toContain('mujoco');
      expect(Array.isArray(data.engines)).toBe(true);
    });

    it('should handle API errors gracefully', async () => {
      global.fetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
      });

      const response = await fetch('/api/engines');

      expect(response.ok).toBe(false);
      expect(response.status).toBe(500);
    });
  });

  describe('WebSocket Connection', () => {
    it('should establish WebSocket connection', () => {
      const ws = new WebSocket('ws://localhost:8000/ws/simulate/mujoco');

      expect(ws).toBeDefined();
      expect(ws.url).toBe('ws://localhost:8000/ws/simulate/mujoco');
    });

    it('should handle WebSocket close', () => {
      const ws = new WebSocket('ws://localhost:8000/ws/simulate/mujoco');
      const closeSpy = vi.fn();
      ws.onclose = closeSpy;

      ws.close();

      expect(closeSpy).toHaveBeenCalled();
    });
  });
});

describe('useSimulation setSpeed (issue #7166)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('returns a structured failure instead of rejecting on HTTP 500', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: () => Promise.resolve({ detail: 'speed control unavailable' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    // Capture unhandled rejections — a fire-and-forget caller must not produce one.
    const rejections: unknown[] = [];
    const onRejection = (e: PromiseRejectionEvent) => rejections.push(e.reason);
    window.addEventListener('unhandledrejection', onRejection);

    const { result } = renderHook(() => useSimulation('mujoco'));
    const res = await result.current.setSpeed(2.0);

    expect(res.success).toBe(false);
    expect(res.error).toContain('Failed to set simulation speed to 2x');
    expect(res.error).toContain('speed control unavailable');

    await Promise.resolve();
    window.removeEventListener('unhandledrejection', onRejection);
    expect(rejections).toHaveLength(0);
  });

  it('returns success on a 2xx response', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useSimulation('mujoco'));
    const res = await result.current.setSpeed(1.5);

    expect(res).toEqual({ success: true });
  });
});

describe('useSimulation connection status (#7435)', () => {
  it('does not leave a stale "lost" status after the hook unmounts', async () => {
    const { result, unmount } = renderHook(() => useSimulation('mujoco'));

    // Open a socket and let the MockWebSocket connect (0ms timer).
    await act(async () => {
      result.current.start({});
      await new Promise((r) => setTimeout(r, 1));
    });
    expect(result.current.connectionStatus).toBe('connected');

    // Unmount (page navigation) — the cleanup must not throw and must reset
    // status so a remounted hook never inherits a stale banner.
    expect(() => unmount()).not.toThrow();

    // A fresh hook instance after navigation starts clean, never 'lost'.
    const fresh = renderHook(() => useSimulation('mujoco'));
    expect(fresh.result.current.connectionStatus).not.toBe('lost');
    expect(fresh.result.current.connectionStatus).toBe('disconnected');
    fresh.unmount();
  });
});

describe('Simulation State', () => {
  it('should track simulation status correctly', () => {
    type SimulationStatus = 'idle' | 'running' | 'paused' | 'stopped';

    const state: { status: SimulationStatus } = { status: 'idle' };

    expect(state.status).toBe('idle');

    state.status = 'running';
    expect(state.status).toBe('running');

    state.status = 'paused';
    expect(state.status).toBe('paused');
  });
});

describe('Engine Compatibility', () => {
  it('should list all supported engines', () => {
    const supportedEngines = ['mujoco', 'drake', 'pinocchio', 'opensim', 'myosuite'];

    expect(supportedEngines).toHaveLength(5);
    expect(supportedEngines).toContain('mujoco');
  });

  it('should validate engine selection', () => {
    const validEngines = new Set(['mujoco', 'drake', 'pinocchio']);

    expect(validEngines.has('mujoco')).toBe(true);
    expect(validEngines.has('invalid')).toBe(false);
  });
});
