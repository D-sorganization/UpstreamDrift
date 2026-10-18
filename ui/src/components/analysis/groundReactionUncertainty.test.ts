import { describe, expect, it } from 'vitest';
import {
  COVARIANCE,
  MAX_COP_MARKERS,
  PER_AXIS,
  copMarkerCovariance,
  copMarkerIndices,
  covarianceEllipse,
  svgEllipse,
  uncertaintyCaption,
  type Covariance,
} from './groundReactionUncertainty';

describe('copMarkerIndices', () => {
  it('keeps only frames with both a CoP and a sigma', () => {
    expect(copMarkerIndices([0.1, null, 0.2], [0.01, 0.01, null], 12)).toEqual([0]);
  });

  it('keeps every usable frame when there are few', () => {
    expect(copMarkerIndices([0, 0, 0, 0, 0], [1, 1, 1, 1, 1], 12)).toEqual([0, 1, 2, 3, 4]);
  });

  it('spaces at most the cap evenly from the first usable frame', () => {
    const picked = copMarkerIndices(Array(100).fill(0), Array(100).fill(0.01), MAX_COP_MARKERS);
    expect(picked.length).toBeLessThanOrEqual(MAX_COP_MARKERS);
    expect(picked[0]).toBe(0);
    expect(copMarkerIndices([], [], 12)).toEqual([]);
  });
});

describe('uncertaintyCaption', () => {
  it('names what is drawn and the declared source', () => {
    expect(uncertaintyCaption(true, COVARIANCE, 'datasheet rev 3')).toBe(
      'Shaded: measured load ±1σ; ellipses: measured CoP ±1σ covariance ' +
        '(declared uncertainty: datasheet rev 3)',
    );
    expect(uncertaintyCaption(false, PER_AXIS, null)).toBe(
      'Ellipses: measured CoP ±1σ per axis (declared uncertainty: not stated)',
    );
    expect(uncertaintyCaption(false, null, 'x')).toBeNull();
  });
});

// The same table is in tests/unit/biomechanics/test_cop_covariance_ellipse.py.
const CASES: Array<[[number, number, number], [number, number, number]]> = [
  [[4e-4, 2.5e-5, 0], [0.02, 0.005, 0]],
  [[2.5e-5, 4e-4, 0], [0.02, 0.005, 90]],
  [[1e-4, 1e-4, 0], [0.01, 0.01, 0]],
  [[2.5e-4, 2.5e-4, 1.5e-4], [0.02, 0.01, 45]],
  [[2.5e-4, 2.5e-4, -1.5e-4], [0.02, 0.01, -45]],
  [[0, 0, 0], [0, 0, 0]],
];

describe('covarianceEllipse', () => {
  it.each(CASES)('matches the Python table for %j', (cov, expected) => {
    const got = covarianceEllipse(...cov);
    expect(got.major).toBeCloseTo(expected[0], 12);
    expect(got.minor).toBeCloseTo(expected[1], 12);
    expect(got.angleDeg).toBeCloseTo(expected[2], 9);
  });

  it('refuses a matrix that is not a covariance', () => {
    expect(() => covarianceEllipse(-1e-4, 1e-4, 0)).toThrow(/covariance/);
    expect(() => covarianceEllipse(1e-4, Number.NaN, 0)).toThrow(/covariance/);
    expect(() => covarianceEllipse(1e-4, 1e-4, 2e-4)).toThrow(/covariance/);
  });
});

describe('copMarkerCovariance', () => {
  const sigma = { x: [0.004], y: [0.001], z: [0.004] };

  it('uses the covariance trace when every term is present', () => {
    const cov = { x: [4e-6], y: [9e-6], z: [-1e-6] };
    expect(copMarkerCovariance(sigma, cov, 0)).toEqual({
      terms: [4e-6, 9e-6, -1e-6],
      kind: COVARIANCE,
    });
  });

  it('falls back to the per-axis sigmas, uncorrelated', () => {
    const partial = { x: [4e-6], y: [null], z: [0] };
    for (const cov of [undefined, partial]) {
      const got = copMarkerCovariance(sigma, cov, 0);
      expect(got?.kind).toBe(PER_AXIS);
      expect(got?.terms[0]).toBeCloseTo(0.004 ** 2, 15);
      expect(got?.terms[1]).toBeCloseTo(0.001 ** 2, 15);
      expect(got?.terms[2]).toBe(0);
    }
  });

  it('falls back to a circle of the worse axis, and to nothing without it', () => {
    const got = copMarkerCovariance({ z: [0.003] }, undefined, 0);
    expect(got?.terms[0]).toBeCloseTo(0.003 ** 2, 15);
    expect(got?.terms[0]).toBe(got?.terms[1]);
    expect(copMarkerCovariance({ x: [0.004], y: [0.001], z: [null] }, undefined, 0)).toBeNull();
  });
});

describe('svgEllipse', () => {
  const rotated = (major: number, minor: number, deg: number): Covariance => {
    const a = (deg * Math.PI) / 180;
    const [c, s] = [Math.cos(a), Math.sin(a)];
    return [
      c * c * major ** 2 + s * s * minor ** 2,
      s * s * major ** 2 + c * c * minor ** 2,
      c * s * (major ** 2 - minor ** 2),
    ];
  };

  it('turns a counter-clockwise data tilt into a negative SVG rotation', () => {
    // SVG y points down, so a +30 deg tilt in data reads as rotate(-30).
    const got = svgEllipse(rotated(0.02, 0.005, 30), 100, 100);
    expect(got.rx).toBeCloseTo(2, 9);
    expect(got.ry).toBeCloseTo(0.5, 9);
    expect(got.rotateDeg).toBeCloseTo(-30, 9);
  });

  it('keeps an axis-aligned ellipse unrotated under unequal scales', () => {
    const got = svgEllipse([0.004 ** 2, 0.001 ** 2, 0], 10, 1000);
    expect(got.rotateDeg).toBe(0);
    expect(got.rx).toBeCloseTo(0.04, 12);
    expect(got.ry).toBeCloseTo(1, 12);
  });

  it('keeps the rotation within (-45, 45] by swapping the semi-axes', () => {
    const got = svgEllipse(rotated(0.02, 0.005, 80), 100, 100);
    expect(got.rotateDeg).toBeCloseTo(10, 9);
    expect(got.rx).toBeCloseTo(0.5, 9);
    expect(got.ry).toBeCloseTo(2, 9);
  });
});
