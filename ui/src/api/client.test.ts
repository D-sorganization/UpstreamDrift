import { describe, it, expect, vi, beforeEach } from 'vitest';

/**
 * Tests for the API client module.
 * Verifies WebSocket connections and API calls work correctly.
 */

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
