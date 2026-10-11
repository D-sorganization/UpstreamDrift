import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';

const { parseAsync } = vi.hoisted(() => ({ parseAsync: vi.fn() }));
vi.mock('three/examples/jsm/loaders/GLTFLoader.js', () => ({
  GLTFLoader: class {
    parseAsync = parseAsync;
  },
}));

import { buildSpecBodyScene, fetchCharacterGltf } from './skinnedBodyPreview';
import type { CharacterSpecRequestBody } from './characterSpecTypes';

const BODY: CharacterSpecRequestBody = {
  preset: 'golfer_pro',
  stature_m: 1.8,
  mass_kg: 80,
  trunk_scale: 1,
  arm_scale: 1,
  shoulder_scale: 1,
  club: 'driver',
};

/** A parsed-glTF stand-in: one 0.4 x 1.8 x 0.3 box centred at (0.2, 0.4, -0.1). */
function fakeGltfScene(): THREE.Group {
  const scene = new THREE.Group();
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.4, 1.8, 0.3));
  mesh.position.set(0.2, 0.4, -0.1);
  scene.add(mesh);
  return scene;
}

describe('fetchCharacterGltf (CMB-7f, #11658)', () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    mockFetch.mockReset();
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('POSTs the spec request body to export/gltf and returns the text', async () => {
    mockFetch.mockResolvedValueOnce({ ok: true, text: () => Promise.resolve('{"asset":{}}') });

    await expect(fetchCharacterGltf(BODY)).resolves.toBe('{"asset":{}}');

    const [url, init] = mockFetch.mock.calls[0];
    expect(url).toMatch(/\/api\/character-builder\/export\/gltf$/);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body)).toEqual(BODY);
  });

  it('surfaces the API detail on failure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      statusText: 'Unprocessable',
      json: () => Promise.resolve({ detail: 'Unknown preset' }),
    });

    await expect(fetchCharacterGltf(BODY)).rejects.toThrow('Unknown preset');
  });

  it('falls back to the HTTP status when the error body is not JSON', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Server Error',
      json: () => Promise.reject(new SyntaxError('not json')),
    });

    await expect(fetchCharacterGltf(BODY)).rejects.toThrow('HTTP 500 Server Error');
  });
});

describe('buildSpecBodyScene (CMB-7f, #11658)', () => {
  beforeEach(() => {
    parseAsync.mockReset();
  });

  it('recolours every mesh and stands the body on the grid', async () => {
    parseAsync.mockResolvedValueOnce({ scene: fakeGltfScene() });

    const scene = await buildSpecBodyScene('{"asset":{"version":"2.0"}}', '#ff0000');

    const meshes: THREE.Mesh[] = [];
    scene.traverse((object) => {
      if (object instanceof THREE.Mesh) meshes.push(object);
    });
    expect(meshes).toHaveLength(1);
    const material = meshes[0].material as THREE.MeshStandardMaterial;
    expect(material.color.getHexString()).toBe('ff0000');
    // Box spans y in [-0.5, 1.3] and is centred at x = 0.2, z = -0.1.
    expect(scene.position.y).toBeCloseTo(0.5, 6);
    expect(scene.position.x).toBeCloseTo(-0.2, 6);
    expect(scene.position.z).toBeCloseTo(0.1, 6);
  });

  it('rejects empty glTF text before parsing', async () => {
    await expect(buildSpecBodyScene('  ', '#ffffff')).rejects.toThrow(TypeError);
    expect(parseAsync).not.toHaveBeenCalled();
  });
});
