import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { fetchEngines, useSimulation } from './client';

/**
 * Tests for the API client module (#8247).
 * Verifies production client logic and hooks for engines and simulation management.
 */

describe('fetchEngines', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should throw when engines is not an array', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ engines: 'not-an-array' }),
      }),
    );

    await expect(fetchEngines()).rejects.toThrow('Unexpected engines response shape');
  });

  it('should throw when engines key is missing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ data: [] }),
      }),
    );

    await expect(fetchEngines()).rejects.toThrow('Unexpected engines response shape');
  });

  it('should return an array of EngineStatus when response is valid', async () => {
    const mockEngines = [
      { name: 'mujoco', available: true, loaded: false, capabilities: [] },
      { name: 'drake', available: true, loaded: true, capabilities: ['autodiff'] },
    ];
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ engines: mockEngines }),
      }),
    );

    const result = await fetchEngines();
    expect(Array.isArray(result)).toBe(true);
    expect(result).toHaveLength(2);
    expect(result[0].name).toBe('mujoco');
    expect(result[1].name).toBe('drake');
  });

  it('should throw "Failed to fetch engines" on HTTP error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
      }),
    );

    await expect(fetchEngines()).rejects.toThrow('Failed to fetch engines');
  });
});

describe('useSimulation lifecycle and commands (#8247)', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('initializes in disconnected state with empty frames', () => {
    const { result } = renderHook(() => useSimulation('mujoco'));
    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.isRunning).toBe(false);
    expect(result.current.isPaused).toBe(false);
    expect(result.current.frames).toEqual([]);
    expect(result.current.currentFrame).toBeNull();
  });

  it('handles pause and resume commands when connected', async () => {
    const { result } = renderHook(() => useSimulation('mujoco'));

    await act(async () => {
      result.current.start({});
      await new Promise((r) => setTimeout(r, 1));
    });

    // Ensure readyState is OPEN for mock socket if needed
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const activeWs = (result as any).current?.wsRef?.current;
    if (activeWs) {
      Object.defineProperty(activeWs, 'readyState', { value: 1, writable: true });
    }

    expect(result.current.connectionStatus).toBe('connected');
    expect(result.current.isRunning).toBe(true);

    act(() => {
      result.current.pause();
    });
    expect(result.current.isPaused).toBe(true);

    act(() => {
      result.current.resume();
    });
    expect(result.current.isPaused).toBe(false);

    act(() => {
      result.current.stop();
    });
    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.isRunning).toBe(false);
  });

  it('returns structured result for setSpeed on HTTP failure', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: () => Promise.resolve({ detail: 'speed control unavailable' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useSimulation('mujoco'));
    const res = await result.current.setSpeed(2.0);

    expect(res.success).toBe(false);
    expect(res.error).toContain('Failed to set simulation speed to 2x');
  });

  it('returns success for setSpeed on 2xx HTTP response', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({}),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useSimulation('mujoco'));
    const res = await result.current.setSpeed(1.5);

    expect(res).toEqual({ success: true });
  });

  it('resets connectionStatus on unmount', async () => {
    const { result, unmount } = renderHook(() => useSimulation('mujoco'));

    await act(async () => {
      result.current.start({});
      await new Promise((r) => setTimeout(r, 1));
    });
    expect(result.current.connectionStatus).toBe('connected');

    unmount();

    const fresh = renderHook(() => useSimulation('mujoco'));
    expect(fresh.result.current.connectionStatus).toBe('disconnected');
    fresh.unmount();
  });
});
