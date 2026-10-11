import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { getApiBase } from '@/api/backend';
import type { CharacterSpecRequestBody } from './characterSpecTypes';

/**
 * Live spec-body preview (CMB-7f, #11658; the CMB-6 #11657 web viewer).
 *
 * The backend's `export/gltf` returns the rigidly skinned body lofted over
 * the compiled spec skeleton (`model_appearance/skinned_body.py`). These
 * helpers fetch it and turn it into a ready-to-mount three.js scene. All
 * mutation (recolour, grounding) happens here, on the freshly parsed scene,
 * so React components never mutate a value they received from a hook.
 */

export const SPEC_BODY_DEFAULT_COLOR = '#4f46e5';

/** POST the spec request body to `export/gltf`; resolve the glTF JSON text. */
export async function fetchCharacterGltf(
  requestBody: CharacterSpecRequestBody,
  signal?: AbortSignal,
): Promise<string> {
  const response = await fetch(`${getApiBase()}/api/character-builder/export/gltf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
    signal,
  });
  if (!response.ok) {
    let detail: string | undefined;
    try {
      detail = ((await response.json()) as { detail?: string }).detail;
    } catch {
      // Body was not JSON — fall through to the generic message below.
    }
    throw new Error(detail ?? `HTTP ${response.status} ${response.statusText}`);
  }
  return response.text();
}

/**
 * Parse glTF text into a scene whose meshes carry `color` and whose feet
 * stand on the y = 0 grid, centred on the vertical axis.
 *
 * Precondition: `gltfText` is non-empty glTF 2.0 JSON.
 */
export async function buildSpecBodyScene(
  gltfText: string,
  color: string,
): Promise<THREE.Group> {
  if (!gltfText.trim()) throw new TypeError('glTF text must be non-empty');
  const gltf = await new GLTFLoader().parseAsync(gltfText, '');
  const scene = gltf.scene;
  scene.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.5,
        metalness: 0.05,
      });
    }
  });
  const box = new THREE.Box3().setFromObject(scene);
  if (!box.isEmpty()) {
    const centre = box.getCenter(new THREE.Vector3());
    scene.position.set(-centre.x, -box.min.y, -centre.z);
  }
  return scene;
}
