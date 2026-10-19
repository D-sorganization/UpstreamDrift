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
      const { grip_style, grip_thumb_offset_deg } = picks;
      expect(visualGripBlock({ grip_style, grip_thumb_offset_deg })).toEqual(
        c.expected.sidecar_grip,
      );
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

  it('refuses out-of-range picks and keeps visual picks out of the physical request', () => {
    expect(() => resolveGrip({ lead_palm_diagonal: 200 })).toThrow(/psi_deg/);
    expect(() => resolveGrip({ hand_spacing: 0 })).toThrow(/positive/);
    expect(() => visualGripBlock({ grip_thumb_offset_deg: 90 })).toThrow(/thumb/);
    expect(hasPhysicalPick({ grip_style: 'interlock' })).toBe(false);
    expect(gripRequestBody({ preset: 'weak', hand_spacing: undefined })).toEqual({ preset: 'weak' });
  });

  it('propagates non-overlap style to physical grip to match Python grip_spec_to_physical', () => {
    const grip = resolveGrip({ grip_style: 'ten_finger' });
    expect(grip.style).toBe('ten_finger');
    const overlapGrip = resolveGrip({ grip_style: 'overlap' });
    expect(overlapGrip.style).toBeUndefined();
  });
});
