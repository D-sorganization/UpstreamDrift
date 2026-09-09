import { describe, expect, it } from 'vitest';
import { segmentForcesAtTime } from './segmentForceFrame';

describe('frame load boundary', () => {
  it('accepts only synchronous, declared signed axial loads', () => {
    const frame = { time_s: 1, source: 'analytical distal section', units: 'N' as const,
      sign_convention: 'tension-positive' as const, values_n: { link: 25 } };
    expect(segmentForcesAtTime(frame, 1)).toEqual({ link: 25 });
    expect(segmentForcesAtTime(frame, 2)).toEqual({});
    expect(segmentForcesAtTime({ ...frame, source: '' }, 1)).toEqual({});
    expect(segmentForcesAtTime(undefined, 1)).toEqual({});
    expect(segmentForcesAtTime({ ...frame, values_n: { link: NaN } }, 1)).toEqual({ link: null });
  });
});
