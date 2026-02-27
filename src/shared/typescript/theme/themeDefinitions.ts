/**
 * Theme Definitions - loaded from the canonical themes.json
 *
 * This module provides the single source of truth for theme colors,
 * shared between Python (PyQt6) and TypeScript (React/Tauri) apps.
 *
 * The themes.json file is the canonical definition; this module
 * transforms snake_case keys to camelCase for TypeScript consumers.
 */

import themesData from '../../theme-definitions/themes.json';

/** Base color keys (14 keys matching PyQt6 THEME_COLOR_KEYS) */
export interface ThemeBaseColors {
  bg: string;
  groupBg: string;
  border: string;
  text: string;
  textSecondary: string;
  label: string;
  focus: string;
  inputBg: string;
  accent: string;
  titleBg: string;
  titleBorder: string;
  tableHeader: string;
  tableAlt: string;
  buttonHover: string;
}

/** Semantic color keys */
export interface ThemeSemanticColors {
  success: string;
  warning: string;
  error: string;
  info: string;
  link: string;
  linkHover: string;
  selectionBg: string;
  selectionText: string;
}

/** Complete theme colors */
export interface ThemeColors extends ThemeBaseColors, ThemeSemanticColors {
  name: string;
}

/** Theme metadata */
export interface ThemeDefinition {
  name: string;
  category: string;
  isDark: boolean;
  colors: ThemeColors;
}

/** Built-in theme IDs */
export type ThemeId =
  | 'light'
  | 'dark'
  | 'slate-gray'
  | 'ocean-blue'
  | 'forest-green'
  | 'monokai'
  | 'dracula'
  | 'one-dark'
  | 'gitpod-dark'
  | 'ms-word'
  | 'ms-excel'
  | 'legal-pad'
  | 'high-contrast';

/** Custom theme stored by user */
export interface CustomTheme extends ThemeColors {
  id: string;
  createdAt: string;
  updatedAt: string;
}

// Key mapping from snake_case (JSON) to camelCase (TypeScript)
const KEY_MAP: Record<string, string> = {
  group_bg: 'groupBg',
  text_secondary: 'textSecondary',
  input_bg: 'inputBg',
  title_bg: 'titleBg',
  title_border: 'titleBorder',
  table_header: 'tableHeader',
  table_alt: 'tableAlt',
  button_hover: 'buttonHover',
  link_hover: 'linkHover',
  selection_bg: 'selectionBg',
  selection_text: 'selectionText',
};

function snakeToCamel(key: string): string {
  return KEY_MAP[key] ?? key;
}

function transformTheme(
  themeDef: (typeof themesData.themes)[keyof typeof themesData.themes],
): ThemeColors {
  const colors: Record<string, string> = { name: themeDef.name };

  // Transform base colors
  for (const [key, value] of Object.entries(themeDef.colors)) {
    colors[snakeToCamel(key)] = value;
  }

  // Transform semantic colors
  for (const [key, value] of Object.entries(themeDef.semantic)) {
    colors[snakeToCamel(key)] = value;
  }

  return colors as unknown as ThemeColors;
}

/** All built-in theme presets, loaded from themes.json */
export const THEME_PRESETS: Record<ThemeId, ThemeColors> = Object.fromEntries(
  Object.entries(themesData.themes).map(([id, def]) => [id, transformTheme(def)]),
) as Record<ThemeId, ThemeColors>;

/** Theme metadata (category, isDark) */
export const THEME_METADATA: Record<ThemeId, { category: string; isDark: boolean }> =
  Object.fromEntries(
    Object.entries(themesData.themes).map(([id, def]) => [
      id,
      { category: def.category, isDark: def.isDark },
    ]),
  ) as Record<ThemeId, { category: string; isDark: boolean }>;

/** Chart colors for data visualization */
export const CHART_COLORS: string[] = themesData.chartColors;

/** Check if a theme is dark */
export function isDarkTheme(theme: ThemeColors): boolean {
  const bgHex = theme.bg.replace('#', '');
  const r = parseInt(bgHex.substring(0, 2), 16);
  const g = parseInt(bgHex.substring(2, 4), 16);
  const b = parseInt(bgHex.substring(4, 6), 16);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
}

/** Calculate WCAG contrast ratio */
export function getContrastRatio(fg: string, bg: string): number {
  const getLuminance = (hex: string): number => {
    const rgb = hex
      .replace('#', '')
      .match(/.{2}/g)
      ?.map((x) => {
        const c = parseInt(x, 16) / 255;
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      }) ?? [0, 0, 0];
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
  };

  const l1 = getLuminance(fg);
  const l2 = getLuminance(bg);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

/** Verify theme readability (WCAG AA = 4.5:1 for normal text) */
export function verifyThemeReadability(theme: ThemeColors): {
  passed: boolean;
  issues: string[];
} {
  const issues: string[] = [];
  const minContrast = 4.5;

  const checks = [
    { fg: theme.text, bg: theme.bg, name: 'text on bg' },
    { fg: theme.text, bg: theme.groupBg, name: 'text on groupBg' },
    { fg: theme.textSecondary, bg: theme.bg, name: 'textSecondary on bg' },
    { fg: theme.label, bg: theme.bg, name: 'label on bg' },
    { fg: theme.text, bg: theme.inputBg, name: 'text on inputBg' },
  ];

  for (const check of checks) {
    const ratio = getContrastRatio(check.fg, check.bg);
    if (ratio < minContrast) {
      issues.push(`${check.name}: ${ratio.toFixed(2)}:1 (needs ${minContrast}:1)`);
    }
  }

  return { passed: issues.length === 0, issues };
}

/** Generate CSS custom properties from theme colors */
export function generateCSSVariables(theme: ThemeColors): Record<string, string> {
  return {
    '--theme-bg': theme.bg,
    '--theme-group-bg': theme.groupBg,
    '--theme-border': theme.border,
    '--theme-text': theme.text,
    '--theme-text-secondary': theme.textSecondary,
    '--theme-label': theme.label,
    '--theme-focus': theme.focus,
    '--theme-input-bg': theme.inputBg,
    '--theme-accent': theme.accent,
    '--theme-title-bg': theme.titleBg,
    '--theme-title-border': theme.titleBorder,
    '--theme-table-header': theme.tableHeader,
    '--theme-table-alt': theme.tableAlt,
    '--theme-button-hover': theme.buttonHover,
    '--theme-success': theme.success,
    '--theme-warning': theme.warning,
    '--theme-error': theme.error,
    '--theme-info': theme.info,
    '--theme-link': theme.link,
    '--theme-link-hover': theme.linkHover,
    '--theme-selection-bg': theme.selectionBg,
    '--theme-selection-text': theme.selectionText,
  };
}

/** Apply theme CSS variables to the document root */
export function applyThemeToDocument(theme: ThemeColors): void {
  const root = document.documentElement;
  const cssVars = generateCSSVariables(theme);

  Object.entries(cssVars).forEach(([key, value]) => {
    root.style.setProperty(key, value);
  });

  if (isDarkTheme(theme)) {
    root.classList.add('dark');
  } else {
    root.classList.remove('dark');
  }
}

/** Apply theme CSS variables to a specific element */
export function applyThemeToElement(
  element: HTMLElement,
  theme: ThemeColors,
): void {
  const cssVars = generateCSSVariables(theme);

  Object.entries(cssVars).forEach(([key, value]) => {
    element.style.setProperty(key, value);
  });

  if (isDarkTheme(theme)) {
    element.classList.add('dark');
  } else {
    element.classList.remove('dark');
  }
}

/** Get all available theme IDs */
export function getThemeIds(): ThemeId[] {
  return Object.keys(THEME_PRESETS) as ThemeId[];
}

/** Get theme display name from ID */
export function getThemeDisplayName(themeId: ThemeId): string {
  return THEME_PRESETS[themeId]?.name ?? themeId;
}
