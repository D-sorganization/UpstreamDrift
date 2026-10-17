import { describe, expect, it } from 'vitest';
import { bandSegments } from './traceGeometry';

// Unit box: t and values map straight onto x in [0, 10] and y in [10, 0].
const box: [number, number, number, number, [number, number], [number, number]] = [
  0,
  10,
  0,
  10,
  [0, 10],
  [0, 10],
];

describe('bandSegments', () => {
  it('closes one polygon: upper bound forward, lower bound back', () => {
    const segs = bandSegments([0, 10], [2, 4], [6, 8], ...box);
    expect(segs).toEqual(['M0.00,4.00L10.00,2.00L10.00,6.00L0.00,8.00Z']);
  });

  it('breaks the band where either bound is unavailable, never bridging it', () => {
    const segs = bandSegments([0, 1, 2, 3, 4], [1, 1, null, 1, 1], [2, 2, 2, 2, 2], ...box);
    expect(segs).toHaveLength(2);
    expect(segs.every((s) => s.endsWith('Z'))).toBe(true);
  });

  it('draws nothing for a lone sample or no data', () => {
    expect(bandSegments([0, 1], [1, null], [2, null], ...box)).toEqual([]);
    expect(bandSegments([], [], [], ...box)).toEqual([]);
  });
});
