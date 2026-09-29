import { apiFetch } from './fetch';

export interface ThemeColors {
  bg: string;
  group_bg: string;
  border: string;
  text: string;
  text_secondary: string;
  label: string;
  focus: string;
  input_bg: string;
  accent: string;
  title_bg: string;
  title_border: string;
  table_header: string;
  table_alt: string;
  button_hover: string;
  success?: string;
  warning?: string;
  error?: string;
  info?: string;
  link?: string;
  link_hover?: string;
  selection_bg?: string;
  selection_text?: string;
}

export interface ActiveThemeResponse {
  name: string;
  is_builtin: boolean;
  colors: ThemeColors;
}

export type SidekickTokenMap = Record<string, string>;

export const SIDEKICK_COLOR_TOKEN_MAP = {
  'sidekick.color.canvas': 'bg',
  'sidekick.color.surface': 'group_bg',
  'sidekick.color.surface.muted': 'table_alt',
  'sidekick.color.surface.raised': 'title_bg',
  'sidekick.color.border': 'border',
  'sidekick.color.border.strong': 'title_border',
  'sidekick.color.text': 'text',
  'sidekick.color.text.muted': 'text_secondary',
  'sidekick.color.text.subtle': 'label',
  'sidekick.color.accent': 'accent',
  'sidekick.color.accent.hover': 'button_hover',
  'sidekick.color.focus': 'focus',
  'sidekick.color.input': 'input_bg',
  'sidekick.color.success': 'success',
  'sidekick.color.warning': 'warning',
  'sidekick.color.error': 'error',
  'sidekick.color.info': 'info',
  'sidekick.color.link': 'link',
  'sidekick.color.link.hover': 'link_hover',
  'sidekick.color.selection': 'selection_bg',
  'sidekick.color.selection.text': 'selection_text',
} as const satisfies Record<string, keyof ThemeColors>;

export const SIDEKICK_FALLBACK_COLOR_TOKENS = {
  'sidekick.color.canvas': '#1a1d23',
  'sidekick.color.surface': '#24272e',
  'sidekick.color.surface.muted': '#24272e',
  'sidekick.color.surface.raised': '#2d3748',
  'sidekick.color.border': '#3a3f4a',
  'sidekick.color.border.strong': '#4a7ba7',
  'sidekick.color.text': '#e1e4e8',
  'sidekick.color.text.muted': '#c9d1d9',
  'sidekick.color.text.subtle': '#8b949e',
  'sidekick.color.accent': '#4a7ba7',
  'sidekick.color.accent.hover': '#5a8fc4',
  'sidekick.color.focus': '#58a6ff',
  'sidekick.color.input': '#0d1117',
  'sidekick.color.success': '#3fb950',
  'sidekick.color.warning': '#d29922',
  'sidekick.color.error': '#f85149',
  'sidekick.color.info': '#58a6ff',
  'sidekick.color.link': '#58a6ff',
  'sidekick.color.link.hover': '#79b8ff',
  'sidekick.color.selection': '#264f78',
  'sidekick.color.selection.text': '#ffffff',
} as const satisfies SidekickTokenMap;

export const SIDEKICK_STATIC_TOKENS = {
  'sidekick.space.1': '4px',
  'sidekick.space.2': '8px',
  'sidekick.space.3': '12px',
  'sidekick.space.4': '16px',
  'sidekick.space.6': '24px',
  'sidekick.space.8': '32px',
  'sidekick.radius.sm': '3px',
  'sidekick.radius.md': '6px',
  'sidekick.radius.lg': '8px',
  'sidekick.radius.chat': '8px',
  'sidekick.border.width': '1px',
  'sidekick.focus.width': '2px',
  'sidekick.font.family': 'Inter, Segoe UI, system-ui, sans-serif',
} as const satisfies SidekickTokenMap;

export async function fetchActiveTheme(): Promise<ActiveThemeResponse> {
  return apiFetch<ActiveThemeResponse>('/api/v1/themes/active');
}

export function sidekickTokenToCSSVariable(tokenName: string): string {
  if (!tokenName.startsWith('sidekick.')) {
    throw new Error(`Unexpected Sidekick token name: ${tokenName}`);
  }
  return `--${tokenName.replace(/\./g, '-')}`;
}

export function buildSidekickTokens(colors: ThemeColors): SidekickTokenMap {
  const fallbackColors: SidekickTokenMap = SIDEKICK_FALLBACK_COLOR_TOKENS;
  return {
    ...Object.fromEntries(
      Object.entries(SIDEKICK_COLOR_TOKEN_MAP).map(([tokenName, themeKey]) => [
        tokenName,
        colors[themeKey] ?? fallbackColors[tokenName],
      ]),
    ),
    ...SIDEKICK_STATIC_TOKENS,
  };
}

export function applyThemeToCSSVariables(colors: ThemeColors) {
  const root = document.documentElement;
  Object.entries(colors).forEach(([key, value]) => {
    // Convert python snake_case to css kebab-case
    const cssKey = key.replace(/_/g, '-');
    root.style.setProperty(`--theme-${cssKey}`, value);
  });

  Object.entries(buildSidekickTokens(colors)).forEach(([tokenName, value]) => {
    root.style.setProperty(sidekickTokenToCSSVariable(tokenName), value);
  });
}
