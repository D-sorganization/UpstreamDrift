/**
 * Settings client — issue #7457 (parity epic #7462).
 *
 * Typed client for the server-side settings endpoints
 * (`GET/PUT /api/v1/settings`) plus the theme list/activate endpoints from
 * the shared theme router. The server file
 * (`~/.upstreamdrift/web_settings.json`) is the source of truth;
 * localStorage is used only as a cache so settings survive browser-storage
 * clears and are shared between Tauri and browser modes.
 *
 * The TypeScript types here mirror the Pydantic `WebSettings` schema in
 * `src/api/routes/settings.py`.
 */

import { apiFetch } from './fetch';
import type { ThemeColors } from './themeClient';

// ── Types (mirror src/api/routes/settings.py) ─────────────────────────────

export type NotificationVerbosity = 'all' | 'errors' | 'silent';

export interface AppearanceSettings {
  theme_id: string;
  font_scale: number;
}

export interface NotificationSettings {
  toast_duration_ms: number;
  verbosity: NotificationVerbosity;
}

export interface SimulationDefaultsSettings {
  default_engine: string;
  duration: number;
  timestep: number;
}

export interface WebSettings {
  appearance: AppearanceSettings;
  notifications: NotificationSettings;
  simulation_defaults: SimulationDefaultsSettings;
}

export const DEFAULT_WEB_SETTINGS: WebSettings = {
  appearance: { theme_id: 'Dark', font_scale: 1.0 },
  notifications: { toast_duration_ms: 4000, verbosity: 'all' },
  simulation_defaults: { default_engine: 'mujoco', duration: 3.0, timestep: 0.002 },
};

// ── Theme listing (shared theme router) ───────────────────────────────────

export interface ThemeDefinition {
  name: string;
  is_builtin: boolean;
  colors: Partial<ThemeColors> & Record<string, string>;
}

export interface ThemeListResponse {
  themes: Record<string, ThemeDefinition>;
}

// ── API calls ──────────────────────────────────────────────────────────────

export async function fetchSettings(): Promise<WebSettings> {
  return apiFetch<WebSettings>('/api/v1/settings');
}

export async function saveSettings(settings: WebSettings): Promise<WebSettings> {
  return apiFetch<WebSettings>('/api/v1/settings', {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}

export async function fetchThemeList(): Promise<ThemeListResponse> {
  return apiFetch<ThemeListResponse>('/api/v1/themes/');
}

export async function setActiveTheme(name: string): Promise<void> {
  await apiFetch<{ success: boolean }>('/api/v1/themes/active', {
    method: 'PUT',
    body: JSON.stringify({ name }),
  });
}

// ── localStorage cache (cache ONLY — server file is the source of truth) ──

export const SETTINGS_CACHE_KEY = 'upstreamdrift.webSettings.v1';

export function readCachedSettings(): WebSettings | null {
  try {
    const raw = window.localStorage.getItem(SETTINGS_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<WebSettings>;
    if (
      typeof parsed !== 'object' ||
      parsed === null ||
      !parsed.appearance ||
      !parsed.notifications ||
      !parsed.simulation_defaults
    ) {
      return null;
    }
    return parsed as WebSettings;
  } catch {
    return null;
  }
}

export function writeCachedSettings(settings: WebSettings): void {
  try {
    window.localStorage.setItem(SETTINGS_CACHE_KEY, JSON.stringify(settings));
  } catch {
    // Storage may be unavailable (private mode / quota) — cache is optional.
  }
}

// ── Local application of settings ──────────────────────────────────────────

/**
 * Apply the font scale as a root CSS variable + root font-size so rem-based
 * layouts scale immediately, mirroring the desktop app-wide zoom.
 */
export function applyFontScale(scale: number): void {
  const clamped = Math.min(2.0, Math.max(0.5, scale));
  const root = document.documentElement;
  root.style.setProperty('--app-font-scale', String(clamped));
  root.style.fontSize = `${clamped * 100}%`;
}

/** Apply everything that takes effect client-side and refresh the cache. */
export function applySettingsLocally(settings: WebSettings): void {
  applyFontScale(settings.appearance.font_scale);
  writeCachedSettings(settings);
}
