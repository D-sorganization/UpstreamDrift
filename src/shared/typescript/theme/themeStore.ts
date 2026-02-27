/**
 * Theme Store Factory
 *
 * Creates a Zustand store for theme management with optional API sync.
 * Each application creates its own store instance with a unique storage key.
 *
 * Usage:
 *   import { createThemeStore } from '@shared/theme';
 *
 *   export const useThemeStore = createThemeStore({
 *     storageKey: 'my-app-theme',
 *     apiBaseUrl: 'http://localhost:8000',
 *   });
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import { THEME_PRESETS, type ThemeColors, type ThemeId } from './themeDefinitions';
import { ThemeApiClient } from './themeApi';

/** Custom theme definition */
export interface CustomTheme {
  name: string;
  colors: ThemeColors;
  createdAt: string;
  modifiedAt: string;
}

/** Per-module theme preference */
export interface ModuleThemePreference {
  themeId: ThemeId | string;
  useGlobal: boolean;
}

/** Theme store state */
export interface ThemeState {
  // Current theme
  currentThemeId: ThemeId | string;
  customThemes: Record<string, CustomTheme>;
  modulePreferences: Record<string, ModuleThemePreference>;

  // Computed
  getCurrentTheme: () => ThemeColors;
  getThemeById: (id: ThemeId | string) => ThemeColors;
  isDarkTheme: () => boolean;
  getAllThemeIds: () => string[];

  // Actions
  setTheme: (themeId: ThemeId | string) => void;
  saveCustomTheme: (name: string, colors: ThemeColors) => void;
  deleteCustomTheme: (name: string) => void;
  setModuleTheme: (moduleId: string, themeId: ThemeId | string, useGlobal: boolean) => void;
  getModuleTheme: (moduleId: string) => ThemeColors;

  // API sync
  syncWithApi: () => Promise<void>;
  pushToApi: () => Promise<void>;

  // CSS variables
  applyThemeToDocument: () => void;
}

/** Store configuration */
export interface ThemeStoreConfig {
  /** Unique localStorage key for this store instance */
  storageKey: string;
  /** Default theme to use */
  defaultTheme?: ThemeId;
  /** API base URL for cross-platform sync (e.g. 'http://localhost:8000') */
  apiBaseUrl?: string;
  /** Re-fetch from API on window focus (default: true when apiBaseUrl is set) */
  syncOnFocus?: boolean;
}

/** Generate CSS custom properties from theme colors */
function generateCSSVariables(theme: ThemeColors): Record<string, string> {
  const vars: Record<string, string> = {};
  for (const [key, value] of Object.entries(theme)) {
    if (key === 'name' || typeof value !== 'string') continue;
    // Convert camelCase to kebab-case: titleBg -> title-bg
    const cssKey = key.replace(/([A-Z])/g, '-$1').toLowerCase();
    vars[`--theme-${cssKey}`] = value;
  }
  return vars;
}

/** Apply CSS variables to the document root */
function applyVarsToDocument(theme: ThemeColors): void {
  const vars = generateCSSVariables(theme);
  const root = document.documentElement;
  for (const [key, value] of Object.entries(vars)) {
    root.style.setProperty(key, value);
  }
  // Set dark mode class
  const isDark = isThemeDark(theme);
  root.classList.toggle('dark', isDark);
  root.setAttribute('data-theme', isDark ? 'dark' : 'light');
}

/** Check if a theme is dark based on background luminance */
function isThemeDark(theme: ThemeColors): boolean {
  const bg = theme.bg;
  if (!bg || !bg.startsWith('#')) return false;
  const r = parseInt(bg.slice(1, 3), 16);
  const g = parseInt(bg.slice(3, 5), 16);
  const b = parseInt(bg.slice(5, 7), 16);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
}

/**
 * Create a themed Zustand store instance.
 *
 * Each application should call this once with its own config:
 *
 * ```ts
 * export const useThemeStore = createThemeStore({
 *   storageKey: 'gasification-theme',
 *   apiBaseUrl: 'http://localhost:8000',
 * });
 * ```
 */
export function createThemeStore(config: ThemeStoreConfig) {
  const {
    storageKey,
    defaultTheme = 'light',
    apiBaseUrl,
    syncOnFocus = !!apiBaseUrl,
  } = config;

  const apiClient = apiBaseUrl ? new ThemeApiClient(apiBaseUrl) : null;

  const store = create<ThemeState>()(
    persist(
      (set, get) => ({
        currentThemeId: defaultTheme,
        customThemes: {},
        modulePreferences: {},

        getCurrentTheme: () => {
          const state = get();
          return state.getThemeById(state.currentThemeId);
        },

        getThemeById: (id: ThemeId | string) => {
          const state = get();
          // Check built-in themes
          if (id in THEME_PRESETS) {
            return THEME_PRESETS[id as ThemeId];
          }
          // Check custom themes
          if (id in state.customThemes) {
            return state.customThemes[id].colors;
          }
          // Fallback to default
          return THEME_PRESETS[defaultTheme] ?? THEME_PRESETS.light;
        },

        isDarkTheme: () => {
          return isThemeDark(get().getCurrentTheme());
        },

        getAllThemeIds: () => {
          const state = get();
          return [
            ...Object.keys(THEME_PRESETS),
            ...Object.keys(state.customThemes),
          ];
        },

        setTheme: (themeId: ThemeId | string) => {
          set({ currentThemeId: themeId });
          // Apply to document immediately
          const theme = get().getThemeById(themeId);
          applyVarsToDocument(theme);
          // Sync with API if available
          if (apiClient) {
            apiClient.setActiveTheme(themeId).catch((err) => {
              console.warn('Failed to sync theme with API:', err);
            });
          }
        },

        saveCustomTheme: (name: string, colors: ThemeColors) => {
          const now = new Date().toISOString();
          set((state) => ({
            customThemes: {
              ...state.customThemes,
              [name]: {
                name,
                colors,
                createdAt: state.customThemes[name]?.createdAt ?? now,
                modifiedAt: now,
              },
            },
          }));
          // Sync with API
          if (apiClient) {
            // Strip 'name' from colors before sending to API
            const { name: _name, ...colorData } = colors;
            apiClient.saveCustomTheme(name, colorData).catch((err) => {
              console.warn('Failed to sync custom theme with API:', err);
            });
          }
        },

        deleteCustomTheme: (name: string) => {
          set((state) => {
            const { [name]: _, ...rest } = state.customThemes;
            return {
              customThemes: rest,
              // If deleted theme was active, switch to default
              currentThemeId:
                state.currentThemeId === name ? defaultTheme : state.currentThemeId,
            };
          });
          if (apiClient) {
            apiClient.deleteCustomTheme(name).catch((err) => {
              console.warn('Failed to sync theme deletion with API:', err);
            });
          }
        },

        setModuleTheme: (
          moduleId: string,
          themeId: ThemeId | string,
          useGlobal: boolean,
        ) => {
          set((state) => ({
            modulePreferences: {
              ...state.modulePreferences,
              [moduleId]: { themeId, useGlobal },
            },
          }));
        },

        getModuleTheme: (moduleId: string) => {
          const state = get();
          const pref = state.modulePreferences[moduleId];
          if (pref && !pref.useGlobal) {
            return state.getThemeById(pref.themeId);
          }
          return state.getCurrentTheme();
        },

        syncWithApi: async () => {
          if (!apiClient) return;
          try {
            // Fetch custom themes from API
            const customThemes = await apiClient.getCustomThemes();
            const now = new Date().toISOString();
            const merged: Record<string, CustomTheme> = {};

            for (const [name, colors] of Object.entries(customThemes)) {
              merged[name] = {
                name,
                colors,
                createdAt: get().customThemes[name]?.createdAt ?? now,
                modifiedAt: now,
              };
            }

            // Also keep any local-only custom themes
            for (const [name, theme] of Object.entries(get().customThemes)) {
              if (!(name in merged)) {
                merged[name] = theme;
              }
            }

            set({ customThemes: merged });

            // Fetch active theme from API
            const active = await apiClient.getActiveTheme();
            if (active.name !== get().currentThemeId) {
              set({ currentThemeId: active.name });
              applyVarsToDocument(get().getCurrentTheme());
            }
          } catch (err) {
            console.warn('Failed to sync with theme API:', err);
          }
        },

        pushToApi: async () => {
          if (!apiClient) return;
          try {
            // Push all local custom themes to API
            for (const [name, theme] of Object.entries(get().customThemes)) {
              const { name: _name, ...colorData } = theme.colors;
              await apiClient.saveCustomTheme(name, colorData);
            }
            // Set active theme
            await apiClient.setActiveTheme(get().currentThemeId);
          } catch (err) {
            console.warn('Failed to push themes to API:', err);
          }
        },

        applyThemeToDocument: () => {
          applyVarsToDocument(get().getCurrentTheme());
        },
      }),
      {
        name: storageKey,
        partialize: (state) => ({
          currentThemeId: state.currentThemeId,
          customThemes: state.customThemes,
          modulePreferences: state.modulePreferences,
        }),
      },
    ),
  );

  // Set up sync-on-focus if API is available
  if (syncOnFocus && apiClient && typeof window !== 'undefined') {
    window.addEventListener('focus', () => {
      store.getState().syncWithApi();
    });

    // Initial sync (non-blocking)
    store.getState().syncWithApi();
  }

  return store;
}

/** Re-export types for convenience */
export type { ThemeColors, ThemeId } from './themeDefinitions';
