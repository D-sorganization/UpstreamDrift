import { describe, expect, it } from 'vitest';
import { defaultForceColorScale, forceColor, validateForceColorScale } from './forceColors';
import examples from '../../../../schemas/force-color-examples.json';

describe('signed axial force colors', () => {
  it('matches the shared Python/web conformance examples', () => {
    for (const entry of examples.cases) {
      const scale = validateForceColorScale({ ...defaultForceColorScale, ...entry.settings });
      expect(forceColor(entry.force_n, '#123456', scale)).toBe(entry.expected);
    }
  });
  it('preserves base colors by default and for unavailable samples', () => {
    expect(forceColor(1000, '#123456', defaultForceColorScale)).toBe('#123456');
    for (const value of [null, undefined, NaN, Infinity]) {
      expect(forceColor(value, '#123456', { ...defaultForceColorScale, enabled: true })).toBe('#123456');
    }
  });
  it('maps tension blue, compression red and zero neutral with clipping', () => {
    const scale = { ...defaultForceColorScale, enabled: true };
    expect(forceColor(1000, '', scale)).toBe('#0000ff');
    expect(forceColor(-1000, '', scale)).toBe('#ff0000');
    expect(forceColor(0, '', scale)).toBe('#ffffff');
    expect(forceColor(5000, '', scale)).toBe('#0000ff');
  });
  it('supports asymmetric ranges, a deadband and custom colors', () => {
    const scale = { ...defaultForceColorScale, enabled: true, tension_limit_n: 110,
      compression_limit_n: 210, deadband_n: 10, tension_color: '#00ff00',
      compression_color: '#ff00ff', neutral_color: '#000000' };
    expect(forceColor(10, '', scale)).toBe('#000000');
    expect(forceColor(60, '', scale)).toBe('#008000');
    expect(forceColor(-110, '', scale)).toBe('#800080');
  });
  it('rejects invalid ranges and colors', () => {
    for (const edit of [{ tension_limit_n: 0 }, { deadband_n: -1 },
      { compression_limit_n: Infinity }, { neutral_color: '#ffffff00' }]) {
      expect(() => validateForceColorScale({ ...defaultForceColorScale, ...edit })).toThrow();
    }
  });
});
