import { describe, expect, it } from 'vitest';
import {
  MAX_COP_MARKERS,
  copMarkerIndices,
  copSemiAxes,
  uncertaintyCaption,
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
    expect(uncertaintyCaption(true, true, 'datasheet rev 3')).toBe(
      'Shaded: measured load ±1σ; ellipses: measured CoP ±1σ per axis ' +
        '(declared uncertainty: datasheet rev 3)',
    );
    expect(uncertaintyCaption(false, true, null)).toBe(
      'Ellipses: measured CoP ±1σ per axis (declared uncertainty: not stated)',
    );
    expect(uncertaintyCaption(false, false, 'x')).toBeNull();
  });
});

describe('copSemiAxes', () => {
  it('uses the per-world-axis sigmas when present', () => {
    expect(copSemiAxes({ x: [0.004], y: [0.001], z: [0.004] }, 0)).toEqual([0.004, 0.001]);
  });

  it('falls back to a circle of the worse axis without them', () => {
    expect(copSemiAxes({ x: [null], y: [null], z: [0.003] }, 0)).toEqual([0.003, 0.003]);
    expect(copSemiAxes({ z: [0.003] }, 0)).toEqual([0.003, 0.003]);
  });

  it('has no ellipse without a worse-axis sigma', () => {
    expect(copSemiAxes({ x: [0.004], y: [0.001], z: [null] }, 0)).toBeNull();
  });
});
