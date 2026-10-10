import { describe, it, expect } from 'vitest';
import {
  appearancePalette,
  rgbaToHex,
  type AppearanceLibrary,
} from './appearanceTypes';

const LIBRARY: AppearanceLibrary = {
  schema_version: 'appearance-v1',
  skin_tones: ['skin_light', 'skin_medium', 'skin_tan', 'skin_dark'],
  clothing: {
    none: { foot: 'shoe_white', hand: 'glove_white' },
    golf_polo_shorts: {
      torso: 'polo_navy',
      pelvis: 'shorts_khaki',
      upper_arm: 'polo_navy',
      shoulder: 'polo_navy',
      thigh: 'shorts_khaki',
      foot: 'shoe_white',
      hand: 'glove_white',
    },
    golf_polo_trousers: {
      torso: 'polo_white',
      pelvis: 'trousers_charcoal',
      upper_arm: 'polo_white',
      shoulder: 'polo_white',
      thigh: 'trousers_charcoal',
      shin: 'trousers_charcoal',
      foot: 'shoe_black',
      hand: 'glove_white',
    },
  },
  club_finishes: ['satin_steel', 'chrome', 'graphite', 'black_pvd'],
  headwear: ['none', 'hair', 'cap'],
  headwear_default_material: { hair: 'hair_brown', cap: 'cap_navy' },
  ground_materials: ['turf', 'studio_floor'],
  materials: {
    skin_medium: { base_color: [0.8, 0.6, 0.46, 1.0], roughness: 0.55, metallic: 0 },
    polo_navy: { base_color: [0.16, 0.27, 0.55, 1.0], roughness: 0.85, metallic: 0 },
    shorts_khaki: { base_color: [0.78, 0.7, 0.52, 1.0], roughness: 0.9, metallic: 0 },
    trousers_charcoal: { base_color: [0.3, 0.32, 0.35, 1.0], roughness: 0.9, metallic: 0 },
  },
};

describe('rgbaToHex', () => {
  it('converts a 0..1 RGBA colour to a 6-digit hex string', () => {
    expect(rgbaToHex([0.8, 0.6, 0.46, 1.0])).toBe('#cc9975');
  });

  it('rounds and clamps out-of-range channels', () => {
    expect(rgbaToHex([0, 0, 0])).toBe('#000000');
    expect(rgbaToHex([1, 1, 1])).toBe('#ffffff');
    expect(rgbaToHex([1.5, -0.5, 0.5])).toBe('#ff0080');
  });
});

describe('appearancePalette', () => {
  it('maps clothing parts to their preset materials', () => {
    const palette = appearancePalette(LIBRARY, {
      skin_tone: 'skin_medium',
      clothing: 'golf_polo_shorts',
      club_finish: 'satin_steel',
      headwear: 'none',
      ground_material: 'turf',
    });

    expect(palette.trunk).toBe(rgbaToHex(LIBRARY.materials.polo_navy.base_color));
    expect(palette.upperArm).toBe(rgbaToHex(LIBRARY.materials.polo_navy.base_color));
    expect(palette.thigh).toBe(rgbaToHex(LIBRARY.materials.shorts_khaki.base_color));
  });

  it('never dresses the forearm — it always shows skin', () => {
    const palette = appearancePalette(LIBRARY, {
      skin_tone: 'skin_medium',
      clothing: 'golf_polo_trousers',
      club_finish: 'satin_steel',
      headwear: 'none',
      ground_material: 'turf',
    });

    const skinHex = rgbaToHex(LIBRARY.materials.skin_medium.base_color);
    expect(palette.forearm).toBe(skinHex);
    expect(palette.head).toBe(skinHex);
  });

  it('falls back to skin tone for a part the preset leaves undressed', () => {
    const palette = appearancePalette(LIBRARY, {
      skin_tone: 'skin_medium',
      clothing: 'none',
      club_finish: 'satin_steel',
      headwear: 'none',
      ground_material: 'turf',
    });

    const skinHex = rgbaToHex(LIBRARY.materials.skin_medium.base_color);
    expect(palette.trunk).toBe(skinHex);
    expect(palette.upperArm).toBe(skinHex);
    expect(palette.thigh).toBe(skinHex);
    expect(palette.shank).toBe(skinHex);
  });

  it('falls back to a default colour when the skin tone material is unknown', () => {
    const palette = appearancePalette(LIBRARY, {
      skin_tone: 'skin_nonexistent',
      clothing: 'none',
      club_finish: 'satin_steel',
      headwear: 'none',
      ground_material: 'turf',
    });

    expect(palette.head).toBe('#cc9966');
    expect(palette.forearm).toBe('#cc9966');
  });
});
