import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import * as THREE from 'three';

const { fetchCharacterGltf, buildSpecBodyScene } = vi.hoisted(() => ({
  fetchCharacterGltf: vi.fn(),
  buildSpecBodyScene: vi.fn(),
}));
vi.mock('./skinnedBodyPreview', () => ({ fetchCharacterGltf, buildSpecBodyScene }));

import { SPEC_BODY_DEBOUNCE_MS, useSpecBodyPreview } from './useSpecBodyPreview';
import type { CharacterSpecRequestBody } from './characterSpecTypes';

const BODY: CharacterSpecRequestBody = {
  preset: null,
  stature_m: 1.8,
  mass_kg: 80,
  trunk_scale: 1,
  arm_scale: 1,
  shoulder_scale: 1,
  club: 'driver',
};

async function flushDebounce() {
  await act(async () => {
    vi.advanceTimersByTime(SPEC_BODY_DEBOUNCE_MS);
  });
  // Let the fetch -> parse promise chain settle.
  await act(async () => {
    for (let tick = 0; tick < 10; tick += 1) await Promise.resolve();
  });
}

describe('useSpecBodyPreview (CMB-7f, #11658)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fetchCharacterGltf.mockReset();
    buildSpecBodyScene.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('does nothing while disabled', async () => {
    const { result } = renderHook(() => useSpecBodyPreview(false, BODY, '#ffffff'));

    await flushDebounce();

    expect(fetchCharacterGltf).not.toHaveBeenCalled();
    expect(result.current).toEqual({ scene: null, loading: false, error: null });
  });

  it('fetches after the debounce and exposes the parsed scene', async () => {
    const scene = new THREE.Group();
    fetchCharacterGltf.mockResolvedValue('{"asset":{}}');
    buildSpecBodyScene.mockResolvedValue(scene);

    const { result } = renderHook(() => useSpecBodyPreview(true, BODY, '#123456'));
    expect(result.current.loading).toBe(true);
    expect(fetchCharacterGltf).not.toHaveBeenCalled();

    await flushDebounce();

    expect(fetchCharacterGltf).toHaveBeenCalledWith(BODY, expect.any(AbortSignal));
    expect(buildSpecBodyScene).toHaveBeenCalledWith('{"asset":{}}', '#123456');
    expect(result.current).toEqual({ scene, loading: false, error: null });
  });

  it('refetches when a slider changes and keeps the old body while loading', async () => {
    const first = new THREE.Group();
    const second = new THREE.Group();
    fetchCharacterGltf.mockResolvedValue('{}');
    buildSpecBodyScene.mockResolvedValueOnce(first).mockResolvedValueOnce(second);

    const { result, rerender } = renderHook(
      ({ body }) => useSpecBodyPreview(true, body, '#ffffff'),
      { initialProps: { body: BODY } },
    );
    await flushDebounce();
    expect(result.current.scene).toBe(first);

    const taller = { ...BODY, stature_m: 1.95 };
    rerender({ body: taller });
    expect(result.current.loading).toBe(true);
    expect(result.current.scene).toBe(first);

    await flushDebounce();
    expect(fetchCharacterGltf).toHaveBeenLastCalledWith(taller, expect.any(AbortSignal));
    expect(result.current).toEqual({ scene: second, loading: false, error: null });
  });

  it('reports the export error', async () => {
    fetchCharacterGltf.mockRejectedValue(new Error('Unknown preset'));

    const { result } = renderHook(() => useSpecBodyPreview(true, BODY, '#ffffff'));
    await flushDebounce();

    expect(result.current).toEqual({ scene: null, loading: false, error: 'Unknown preset' });
  });
});
