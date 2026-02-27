/**
 * Theme API Client
 *
 * TypeScript client for the shared theme REST API.
 * Works with both Tauri and pure web applications.
 */

import type { ThemeColors } from './themeDefinitions';

/** API response for theme lists */
export interface ThemeListResponse {
  themes: Record<string, {
    name: string;
    is_builtin: boolean;
    colors: Record<string, string>;
  }>;
}

/** API response for active theme */
export interface ActiveThemeResponse {
  name: string;
  is_builtin: boolean;
  colors: Record<string, string>;
}

/** API response for theme operations */
export interface ThemeOperationResponse {
  success: boolean;
  message: string;
  theme_name: string | null;
}

/** Snake-to-camel case key mapping */
const SNAKE_TO_CAMEL: Record<string, string> = {
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

/** Camel-to-snake case key mapping */
const CAMEL_TO_SNAKE: Record<string, string> = Object.fromEntries(
  Object.entries(SNAKE_TO_CAMEL).map(([k, v]) => [v, k])
);

/** Transform API (snake_case) colors to TypeScript (camelCase) */
function apiColorsToTs(colors: Record<string, string>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(colors)) {
    result[SNAKE_TO_CAMEL[key] ?? key] = value;
  }
  return result;
}

/** Transform TypeScript (camelCase) colors to API (snake_case) */
function tsColorsToApi(colors: Record<string, string>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(colors)) {
    result[CAMEL_TO_SNAKE[key] ?? key] = value;
  }
  return result;
}

/**
 * Theme API client for communicating with the backend theme endpoints.
 *
 * Usage:
 *   const client = new ThemeApiClient('http://localhost:8000');
 *   const themes = await client.getBuiltinThemes();
 */
export class ThemeApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    // Remove trailing slash
    this.baseUrl = baseUrl.replace(/\/+$/, '');
  }

  private get url(): string {
    return `${this.baseUrl}/api/v1/themes`;
  }

  /** List all built-in themes (camelCase colors). */
  async getBuiltinThemes(): Promise<Record<string, ThemeColors>> {
    const resp = await fetch(`${this.url}/builtin`);
    if (!resp.ok) throw new Error(`Failed to fetch builtin themes: ${resp.statusText}`);
    const data: ThemeListResponse = await resp.json();
    return this.transformThemeList(data);
  }

  /** List all custom themes (camelCase colors). */
  async getCustomThemes(): Promise<Record<string, ThemeColors>> {
    const resp = await fetch(`${this.url}/custom`);
    if (!resp.ok) throw new Error(`Failed to fetch custom themes: ${resp.statusText}`);
    const data: ThemeListResponse = await resp.json();
    return this.transformThemeList(data);
  }

  /** List all themes (built-in + custom). */
  async getAllThemes(): Promise<Record<string, ThemeColors>> {
    const resp = await fetch(this.url);
    if (!resp.ok) throw new Error(`Failed to fetch themes: ${resp.statusText}`);
    const data: ThemeListResponse = await resp.json();
    return this.transformThemeList(data);
  }

  /** Get the currently active theme. */
  async getActiveTheme(): Promise<{ name: string; isBuiltin: boolean; colors: ThemeColors }> {
    const resp = await fetch(`${this.url}/active`);
    if (!resp.ok) throw new Error(`Failed to fetch active theme: ${resp.statusText}`);
    const data: ActiveThemeResponse = await resp.json();
    const camelColors = apiColorsToTs(data.colors);
    return {
      name: data.name,
      isBuiltin: data.is_builtin,
      colors: { name: data.name, ...camelColors } as ThemeColors,
    };
  }

  /** Set the active theme on the backend. */
  async setActiveTheme(name: string): Promise<ThemeOperationResponse> {
    const resp = await fetch(`${this.url}/active`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!resp.ok) throw new Error(`Failed to set active theme: ${resp.statusText}`);
    return resp.json();
  }

  /** Save a custom theme. Colors should use camelCase keys. */
  async saveCustomTheme(
    name: string,
    colors: Record<string, string>,
    apply = false,
  ): Promise<ThemeOperationResponse> {
    const resp = await fetch(`${this.url}/custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        colors: tsColorsToApi(colors),
        apply,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail ?? `Failed to save custom theme: ${resp.statusText}`);
    }
    return resp.json();
  }

  /** Delete a custom theme by name. */
  async deleteCustomTheme(name: string): Promise<ThemeOperationResponse> {
    const resp = await fetch(`${this.url}/custom/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    });
    if (!resp.ok) throw new Error(`Failed to delete custom theme: ${resp.statusText}`);
    return resp.json();
  }

  /** Transform API response to camelCase ThemeColors map. */
  private transformThemeList(data: ThemeListResponse): Record<string, ThemeColors> {
    const result: Record<string, ThemeColors> = {};
    for (const [key, theme] of Object.entries(data.themes)) {
      const camelColors = apiColorsToTs(theme.colors);
      result[key] = { name: theme.name, ...camelColors } as ThemeColors;
    }
    return result;
  }
}
