import { describe, expect, it } from 'vitest';
import { BALL_RADIUS_M, ballPositionAtAddress } from './ballGeometry';

describe('ballPositionAtAddress (mirrors model_appearance/ball.py, Y up)', () => {
  it('uses the regulation radius', () => {
    expect(BALL_RADIUS_M).toBeCloseTo(0.021335, 9);
  });

  it('touches the face centre horizontally and rests on the ground', () => {
    const p = ballPositionAtAddress([0.5, 0.03, 0.1], [1, 0, 0], 0);
    expect(p[0]).toBeCloseTo(0.5 + BALL_RADIUS_M, 9);
    expect(p[1]).toBeCloseTo(BALL_RADIUS_M, 9);
    expect(p[2]).toBeCloseTo(0.1, 9);
  });

  it('discards the vertical component of a lofted normal and respects ground height', () => {
    const n = [Math.cos(0.3), Math.sin(0.3), 0] as [number, number, number];
    const p = ballPositionAtAddress([0, 0.02, 0], n, 0.5);
    expect(p[1]).toBeCloseTo(0.5 + BALL_RADIUS_M, 9);
    expect(p[0]).toBeCloseTo(BALL_RADIUS_M * Math.cos(0.3), 9);
  });

  it('rejects bad input', () => {
    expect(() => ballPositionAtAddress([0, 0, 0], [0, 0, 0], 0)).toThrow(/nonzero/);
    expect(() => ballPositionAtAddress([NaN, 0, 0], [1, 0, 0], 0)).toThrow(/finite/);
    expect(() => ballPositionAtAddress([0, 0, 0], [1, 0, 0], 0, -1)).toThrow(/radius/);
  });
});
