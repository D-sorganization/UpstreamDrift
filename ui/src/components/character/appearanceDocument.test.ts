import { describe, it, expect } from 'vitest';
import { customizeDocument, hexToRgb, isHex } from './appearanceDocument';
import type { AppearanceLibrary } from './appearanceTypes';
import fixture from '../../../../tests/fixtures/appearance/colour_roundtrip.json';

const LIBRARY = {
  ...fixture.library,
  skin_tones: [],
  club_finishes: [],
  headwear: ['none', 'hair', 'cap'],
  ground_materials: [],
} as unknown as AppearanceLibrary;

describe('customizeDocument', () => {
  for (const c of fixture.cases) {
    it(`matches the Python writer: ${c.name}`, () => {
      const out = customizeDocument(LIBRARY, c.picks as never);
      expect(out.skin_tone).toBe(c.expected.skin_tone);
      expect(out.glove).toBe(c.expected.glove);
      expect(out.headwear_material ?? null).toBe(c.expected.headwear_material);
      const colours = Object.fromEntries(
        Object.entries(out.materials).map(([k, v]) => [k, v.base_color]),
      );
      expect(colours).toEqual(c.expected.materials);
    });
  }

  it('defaults to bare hands and rejects an unknown glove', () => {
    const base = { skin_tone: 'skin_2', clothing: 'golf_polo_trousers', headwear: 'hair' };
    expect(customizeDocument(LIBRARY, base).glove).toBe('none');
    expect(() => customizeDocument(LIBRARY, { ...base, glove: 'left' as never })).toThrow(/glove/);
  });

  it('keeps roughness and texture of the recoloured garment', () => {
    const out = customizeDocument(LIBRARY, {
      skin_tone: 'skin_2',
      clothing: 'golf_polo_trousers',
      headwear: 'hair',
      top_color: '#aa0000',
    });
    expect(out.materials['polo_white']!.roughness).toBe(LIBRARY.materials['polo_white']!.roughness);
  });

  it('rejects a hair colour on a cap and a garment colour with no garment', () => {
    expect(() =>
      customizeDocument(LIBRARY, {
        skin_tone: 'skin_2',
        clothing: 'golf_polo_trousers',
        headwear: 'cap',
        hair_color: '#112233',
      }),
    ).toThrow(/headwear/);
    expect(() =>
      customizeDocument(LIBRARY, {
        skin_tone: 'skin_2',
        clothing: 'none',
        headwear: 'hair',
        top_color: '#112233',
      }),
    ).toThrow(/no top/);
  });
});

describe('hex helpers', () => {
  it('parses #rrggbb to byte/255 and rejects other text', () => {
    expect(hexToRgb('#FF8000')).toEqual([1, 128 / 255, 0]);
    expect(isHex('red')).toBe(false);
    expect(() => hexToRgb('#12345')).toThrow(/hex/);
  });
});
