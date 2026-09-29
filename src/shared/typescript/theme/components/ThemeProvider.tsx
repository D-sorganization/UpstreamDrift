/**
 * Theme Provider Component
 *
 * Wraps an application with theme context and auto-applies CSS variables.
 * Works with the shared createThemeStore factory.
 */

import React, { createContext, useContext, useEffect, type ReactNode } from 'react';
import type { ThemeColors, ThemeId } from '../themeDefinitions';

/** Theme context value */
export interface ThemeContextValue {
  theme: ThemeColors;
  themeId: ThemeId | string;
  isDark: boolean;
  setTheme: (id: ThemeId | string) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

/** Hook to access the current theme from context */
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return ctx;
}

/** Props for ThemeProvider */
export interface ThemeProviderProps {
  /** Zustand store hook (e.g. useThemeStore) */
  useStore: () => {
    currentThemeId: ThemeId | string;
    getCurrentTheme: () => ThemeColors;
    isDarkTheme: () => boolean;
    setTheme: (id: ThemeId | string) => void;
    applyThemeToDocument: () => void;
  };
  children: ReactNode;
}

/**
 * Theme provider that wraps an application with theme context
 * and auto-applies CSS variables to the document.
 *
 * Usage:
 *   <ThemeProvider useStore={useThemeStore}>
 *     <App />
 *   </ThemeProvider>
 */
export function ThemeProvider({ useStore, children }: ThemeProviderProps) {
  const store = useStore();

  // Apply theme CSS variables whenever theme changes
  useEffect(() => {
    store.applyThemeToDocument();
  }, [store.currentThemeId]);

  const value: ThemeContextValue = {
    theme: store.getCurrentTheme(),
    themeId: store.currentThemeId,
    isDark: store.isDarkTheme(),
    setTheme: store.setTheme,
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}
