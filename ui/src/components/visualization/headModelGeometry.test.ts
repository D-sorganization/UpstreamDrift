import { describe, expect, it } from 'vitest';
import {
  DEFAULT_HEAD_LENGTH_M,
  SKULL_CENTRE_Z,
  buildHeadParts,
  canonicalToScene,
  domeMesh,
  meshVolume,
  skullMesh,
  skullPoint,
} from './headModelGeometry';

const L = DEFAULT_HEAD_LENGTH_M;
const byName = (name: string, headwear: 'none' | 'hair' | 'cap' = 'hair') => {
  const part = buildHeadParts(L, headwear).find((p) => p.name === name);
  if (!part) throw new Error(`missing part ${name}`);
  return part;
};

describe('buildHeadParts', () => {
  it('has skull, eyes, nose, ears, neck and hair', () => {
    const names = buildHeadParts(L).map((p) => p.name);
    for (const n of [
      'skull', 'neck', 'nose', 'mouth', 'ear_l', 'ear_r', 'eye_l', 'eye_r',
      'iris_l', 'iris_r', 'brow_l', 'brow_r', 'hair',
    ]) {
      expect(names).toContain(n);
    }
  });

  it('supports a cap with a visor and a bare head', () => {
    const cap = buildHeadParts(L, 'cap').map((p) => p.name);
    expect(cap).toEqual(expect.arrayContaining(['cap', 'visor']));
    const bare = buildHeadParts(L, 'none').map((p) => p.name);
    expect(bare).not.toContain('hair');
    expect(bare).not.toContain('cap');
  });

  it('puts the face forward: eyes, nose and visor are ahead of the skull centre', () => {
    for (const n of ['eye_l', 'eye_r', 'nose', 'mouth']) {
      expect(byName(n).centre[0]).toBeGreaterThan(0.2 * L);
    }
    expect(byName('visor', 'cap').centre[0]).toBeGreaterThan(0.3 * L);
    // Ears sit at the sides and slightly behind the centre.
    expect(byName('ear_l').centre[0]).toBeLessThan(0);
    expect(Math.abs(byName('ear_l').centre[1])).toBeGreaterThan(0.2 * L);
  });

  it('keeps the eyes above the nose, the nose above the mouth', () => {
    expect(byName('eye_l').centre[2]).toBeGreaterThan(byName('nose').centre[2]);
    expect(byName('nose').centre[2]).toBeGreaterThan(byName('mouth').centre[2]);
  });

  it('mirrors left and right about y = 0', () => {
    for (const base of ['eye', 'iris', 'brow', 'ear']) {
      const l = byName(`${base}_l`);
      const r = byName(`${base}_r`);
      expect(l.centre[0]).toBeCloseTo(r.centre[0], 9);
      expect(l.centre[1]).toBeCloseTo(-r.centre[1], 9);
      expect(l.centre[2]).toBeCloseTo(r.centre[2], 9);
    }
    expect(byName('nose').centre[1]).toBeCloseTo(0, 9);
  });

  it('rejects a non-positive head length', () => {
    expect(() => buildHeadParts(0)).toThrow();
    expect(() => buildHeadParts(Number.NaN)).toThrow();
  });
});

describe('skull geometry', () => {
  it('has its vertex one head length above the neck point', () => {
    expect(skullPoint([0, 0, 1], L)[2]).toBeCloseTo(L, 9);
    expect(skullPoint([0, 0, -1], L)[2]).toBeCloseTo((SKULL_CENTRE_Z - 0.42) * L, 9);
  });

  it('narrows toward the chin', () => {
    const wide = skullPoint([0, 1, 0], L)[1];
    const jaw = skullPoint([0, Math.SQRT1_2, -Math.SQRT1_2], L)[1] / Math.SQRT1_2;
    expect(jaw).toBeLessThan(wide);
  });

  it('is a closed mesh with outward winding', () => {
    expect(meshVolume(skullMesh(L))).toBeGreaterThan(0);
    expect(meshVolume(domeMesh(L, 0.95, 1.9, 1.05))).toBeGreaterThan(0);
  });
});

describe('canonicalToScene', () => {
  it('maps forward to +x, up to +y and left to -z', () => {
    expect(canonicalToScene([1, 0, 0])).toEqual([1, 0, 0]);
    expect(canonicalToScene([0, 0, 1])).toEqual([0, 1, 0]);
    expect(canonicalToScene([0, 1, 0])).toEqual([0, 0, -1]);
  });
});
