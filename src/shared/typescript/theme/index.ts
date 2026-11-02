/**
 * Shared Theme Package
 *
 * Provides cross-platform theme management for React/Tauri applications,
 * backed by the canonical themes.json and optional REST API sync.
 */

// Theme definitions from JSON source of truth
export {
  CHART_COLORS,
  LEGACY_THEME_IDS,
  THEME_METADATA,
  THEME_PRESETS,
  type ThemeBaseColors,
  type ThemeColors,
  type ThemeDefinition,
  type ThemeId,
  type ThemeSemanticColors,
  applyThemeToDocument,
  applyThemeToElement,
  applySidekickThemeToElement,
  generateCSSVariables,
  generateSidekickCSSVariables,
  getContrastRatio,
  isDarkTheme,
  resolveThemeId,
  verifyThemeReadability,
} from './themeDefinitions';

// API client for REST theme sync
export { ThemeApiClient } from './themeApi';
export type {
  ActiveThemeResponse,
  ThemeListResponse,
  ThemeOperationResponse,
} from './themeApi';

// Store factory
export {
  createThemeStore,
  type CustomTheme,
  type ModuleThemePreference,
  type ThemeState,
  type ThemeStoreConfig,
} from './themeStore';
