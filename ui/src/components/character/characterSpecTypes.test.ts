import { describe, it, expect } from 'vitest';
import { DEFAULT_PARAMS, specRequestBody } from './characterSpecTypes';

describe('specRequestBody', () => {
  it('maps an empty preset id to preset: null', () => {
    expect(specRequestBody('', DEFAULT_PARAMS).preset).toBeNull();
  });

  it('keeps a non-empty preset id', () => {
    expect(specRequestBody('golfer_pro', DEFAULT_PARAMS).preset).toBe(
      'golfer_pro',
    );
  });

  it('copies all six slider fields exactly', () => {
    const params = {
      stature_m: 1.9,
      mass_kg: 88,
      trunk_scale: 1.1,
      arm_scale: 1.05,
      shoulder_scale: 1.2,
      club: 'iron7',
    };

    expect(specRequestBody('', params)).toEqual({
      preset: null,
      stature_m: 1.9,
      mass_kg: 88,
      trunk_scale: 1.1,
      arm_scale: 1.05,
      shoulder_scale: 1.2,
      club: 'iron7',
    });
  });
});
