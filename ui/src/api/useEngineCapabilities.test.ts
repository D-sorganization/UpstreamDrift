/**
 * Tests for useEngineCapabilities hook.
 *
 * See issue #1204
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { useEngineCapabilities } from './useEngineCapabilities';
import type { EngineCapabilitiesData } from './useEngineCapabilities';

const mockCapabilities: EngineCapabilitiesData = {
  engine_name: 'MuJoCo',
  engine_type: 'mujoco',
  capabilities: [
    { name: 'mass_matrix', level: 'full', supported: true },
    { name: 'jacobian', level: 'full', supported: true },
    { name: 'contact_forces', level: 'partial', supported: true },
    { name: 'video_export', level: 'none', supported: false },
  ],
  summary: { full: 2, partial: 1, none: 1 },
};

describe('useEngineCapabilities', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('starts in idle state when no engine type provided', () => {
    const { result } = renderHook(() => useEngineCapabilities());

    expect(result.current.capabilities).toBeNull();
    expect(result.current.loadState).toBe('idle');
    expect(result.current.error).toBeNull();
  });

  it('fetches capabilities when engine type is provided', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockCapabilities),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useEngineCapabilities('mujoco'));

    await waitFor(() => {
      expect(result.current.loadState).toBe('loaded');
    });

    expect(result.current.capabilities).toEqual(mockCapabilities);
    expect(fetchMock).toHaveBeenCalledWith('/engines/mujoco/capabilities');
  });

  it('handles fetch errors', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      json: () => Promise.resolve({ detail: 'Engine not found' }),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useEngineCapabilities('unknown'));

    await waitFor(() => {
      expect(result.current.loadState).toBe('error');
    });

    expect(result.current.error).toBe('Engine not found');
  });

  it('isSupported returns true for full and partial capabilities', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockCapabilities),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useEngineCapabilities('mujoco'));

    await waitFor(() => {
      expect(result.current.loadState).toBe('loaded');
    });

    expect(result.current.isSupported('mass_matrix')).toBe(true);
    expect(result.current.isSupported('contact_forces')).toBe(true);
    expect(result.current.isSupported('video_export')).toBe(false);
    expect(result.current.isSupported('nonexistent')).toBe(false);
  });

  it('isFullySupported returns true only for full level', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockCapabilities),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useEngineCapabilities('mujoco'));

    await waitFor(() => {
      expect(result.current.loadState).toBe('loaded');
    });

    expect(result.current.isFullySupported('mass_matrix')).toBe(true);
    expect(result.current.isFullySupported('contact_forces')).toBe(false);
  });

  it('getLevel returns correct levels', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockCapabilities),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result } = renderHook(() => useEngineCapabilities('mujoco'));

    await waitFor(() => {
      expect(result.current.loadState).toBe('loaded');
    });

    expect(result.current.getLevel('mass_matrix')).toBe('full');
    expect(result.current.getLevel('contact_forces')).toBe('partial');
    expect(result.current.getLevel('video_export')).toBe('none');
    expect(result.current.getLevel('nonexistent')).toBe('none');
  });

  it('resets to idle when engine type changes to undefined', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockCapabilities),
    });
    vi.stubGlobal('fetch', fetchMock);

    const { result, rerender } = renderHook(
      ({ engineType }: { engineType?: string }) => useEngineCapabilities(engineType),
      { initialProps: { engineType: 'mujoco' } as { engineType?: string } },
    );

    await waitFor(() => {
      expect(result.current.loadState).toBe('loaded');
    });

    rerender({ engineType: undefined });

    expect(result.current.capabilities).toBeNull();
    expect(result.current.loadState).toBe('idle');
  });
});
