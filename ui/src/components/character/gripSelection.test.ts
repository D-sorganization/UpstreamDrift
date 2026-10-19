import { describe, it, expect } from 'vitest';
import {
  GRIP_PRESETS,
  gripRequestBody,
  hasPhysicalPick,
  resolveGrip,
  visualGripBlock,
  type GripPicks,
} from './gripSelection';
import fixture from '../../../../tests/fixtures/grip/grip_roundtrip.json';

describe('resolveGrip', () => {
  for (const c of fixture.cases) {
    it(`matches the Python resolver: ${c.name}`, () => {
      const picks = c.picks as GripPicks;
      expect(hasPhysicalPick(picks)).toBe(c.expected.refit_required);
      if (c.expected.grip !== null) {
        expect(resolveGrip(picks)).toEqual(c.expected.grip);
      }
      const {
        grip_thumb_offset_deg,
        grip_strength_deg,
        lead_roll_offset_deg,
        trail_roll_offset_deg,
      } = picks;
      expect(
        visualGripBlock({
          grip_thumb_offset_deg,
          grip_strength_deg,
          lead_roll_offset_deg,
          trail_roll_offset_deg,
        }),
      ).toEqual(c.expected.sidecar_grip);
    });
  }

  it('mirrors the strength presets of the Python contract', () => {
    expect(GRIP_PRESETS).toEqual({ weak: -15, neutral: 0, strong: 15 });
  });

  it('lets an explicit pick win over the preset, and the document under both', () => {
    const base = resolveGrip({ preset: 'strong', hand_spacing: 0.09 });
    const next = resolveGrip({ preset: 'weak', lead_shaft_rotation: 3 }, base);
    expect(next.lead.alpha_deg).toBe(3);
    expect(next.trail.alpha_deg).toBe(-15);
    expect(next.hand_spacing_m).toBe(0.09);
    expect(resolveGrip({}, base)).toEqual(base);
  });

  it('refuses out-of-range picks and classifies grip_style as physical while keeping HRG-A2 picks visual', () => {
    expect(() => resolveGrip({ lead_palm_diagonal: 200 })).toThrow(/psi_deg/);
    expect(() => resolveGrip({ hand_spacing: 0 })).toThrow(/positive/);
    expect(() => visualGripBlock({ grip_thumb_offset_deg: 90 })).toThrow(/thumb/);
    expect(() => visualGripBlock({ grip_strength_deg: 45 })).toThrow(/grip_strength_deg/);
    expect(() => visualGripBlock({ lead_roll_offset_deg: -45 })).toThrow(/lead_roll_offset_deg/);
    expect(() => visualGripBlock({ trail_roll_offset_deg: 45 })).toThrow(/trail_roll_offset_deg/);
    // HRG-GS-G1 (#13346): grip_style is physical; HRG-A2 (#13166) picks stay visual.
    expect(hasPhysicalPick({ grip_style: 'interlock' })).toBe(true);
    expect(hasPhysicalPick({ grip_style: 'ten_finger' })).toBe(true);
    expect(hasPhysicalPick({ grip_style: 'overlap' })).toBe(false);
    expect(hasPhysicalPick({ grip_thumb_offset_deg: 20 })).toBe(false);
    expect(
      hasPhysicalPick({
        grip_strength_deg: 10,
        lead_roll_offset_deg: -5,
        trail_roll_offset_deg: 8,
      }),
    ).toBe(false);
    expect(gripRequestBody({ preset: 'weak', hand_spacing: undefined })).toEqual({ preset: 'weak' });
  });
});
