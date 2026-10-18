import { describe, expect, it } from 'vitest';
import {
  MAX_COP_MARKERS,
  copMarkerIndices,
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
      'Shaded: measured load ±1σ; circles: measured CoP ±1σ per axis ' +
        '(declared uncertainty: datasheet rev 3)',
    );
    expect(uncertaintyCaption(false, true, null)).toBe(
      'Circles: measured CoP ±1σ per axis (declared uncertainty: not stated)',
    );
    expect(uncertaintyCaption(false, false, 'x')).toBeNull();
  });
});
