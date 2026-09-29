import { describe, expect, it } from 'vitest';

import {
  applyThemeToCSSVariables,
  buildSidekickTokens,
  sidekickTokenToCSSVariable,
  type ThemeColors,
} from './themeClient';

const colors: ThemeColors = {
  bg: '#000001',
  group_bg: '#000002',
  border: '#000003',
  text: '#000004',
  text_secondary: '#000005',
  label: '#000006',
  focus: '#000007',
  input_bg: '#000008',
  accent: '#000009',
  title_bg: '#00000a',
  title_border: '#00000b',
  table_header: '#00000c',
  table_alt: '#00000d',
  button_hover: '#00000e',
};

describe('themeClient Sidekick adapter', () => {
  it('maps launcher theme colors to canonical Sidekick tokens', () => {
    const tokens = buildSidekickTokens(colors);

    expect(tokens['sidekick.color.canvas']).toBe('#000001');
    expect(tokens['sidekick.color.surface']).toBe('#000002');
    expect(tokens['sidekick.color.text']).toBe('#000004');
    expect(tokens['sidekick.color.focus']).toBe('#000007');
    expect(tokens['sidekick.color.accent.hover']).toBe('#00000e');
    expect(tokens['sidekick.radius.chat']).toBe('8px');
  });

  it('converts token names to CSS custom properties', () => {
    expect(sidekickTokenToCSSVariable('sidekick.color.surface.muted')).toBe(
      '--sidekick-color-surface-muted',
    );
    expect(() => sidekickTokenToCSSVariable('theme.color.surface')).toThrow(
      /Unexpected Sidekick token/,
    );
  });

  it('applies legacy theme variables and Sidekick variables together', () => {
    applyThemeToCSSVariables(colors);

    expect(
      document.documentElement.style.getPropertyValue('--theme-group-bg'),
    ).toBe('#000002');
    expect(
      document.documentElement.style.getPropertyValue(
        '--sidekick-color-surface',
      ),
    ).toBe('#000002');
    expect(
      document.documentElement.style.getPropertyValue('--sidekick-space-4'),
    ).toBe('16px');
  });
});
