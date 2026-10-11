import { useEffect, useState } from 'react';
import type * as THREE from 'three';
import { errorMessage, type CharacterSpecRequestBody } from './characterSpecTypes';
import { buildSpecBodyScene, fetchCharacterGltf } from './skinnedBodyPreview';

/** Debounce between a slider edit and the glTF refetch, in milliseconds. */
export const SPEC_BODY_DEBOUNCE_MS = 400;

export interface SpecBodyPreviewState {
  scene: THREE.Group | null;
  loading: boolean;
  error: string | null;
}

interface Loaded {
  key: string;
  scene: THREE.Group | null;
  error: string | null;
}

/**
 * Fetch and parse the spec body's glTF whenever the spec request body or
 * colour changes (debounced), while `enabled` (CMB-7f, #11658).
 *
 * State is only set from the debounced async callback; `loading` is derived
 * by comparing the loaded key with the current one, so the effect never
 * sets state synchronously.
 */
export function useSpecBodyPreview(
  enabled: boolean,
  requestBody: CharacterSpecRequestBody,
  color: string,
): SpecBodyPreviewState {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const key = `${JSON.stringify(requestBody)}|${color}`;

  useEffect(() => {
    if (!enabled) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      fetchCharacterGltf(requestBody, controller.signal)
        .then((text) => buildSpecBodyScene(text, color))
        .then((scene) => {
          if (!controller.signal.aborted) setLoaded({ key, scene, error: null });
        })
        .catch((err: unknown) => {
          if (!controller.signal.aborted) {
            setLoaded({ key, scene: null, error: errorMessage(err) });
          }
        });
    }, SPEC_BODY_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [enabled, key, requestBody, color]);

  if (!enabled) return { scene: null, loading: false, error: null };
  const current = loaded?.key === key ? loaded : null;
  return {
    // Keep showing the previous body while the next one loads.
    scene: current?.scene ?? loaded?.scene ?? null,
    loading: current === null,
    error: current?.error ?? null,
  };
}
