/**
 * Web-settings bootstrap hook — issue #7457.
 *
 * Called once from <App/>. Applies the cached settings immediately (font
 * scale) so there is no flash, then fetches the authoritative server
 * settings, re-applies them, refreshes the cache, and hydrates the
 * simulation store defaults. Hydration is guarded inside the store so an
 * in-session parameter change is never clobbered (#7424).
 */

import { useEffect } from 'react';
import {
  applyFontScale,
  applySettingsLocally,
  fetchSettings,
  readCachedSettings,
  type WebSettings,
} from './settingsClient';
import { useSimulationStore } from '@/stores/useSimulationStore';

function hydrateSimulationDefaults(settings: WebSettings): void {
  useSimulationStore.getState().hydrateDefaults({
    duration: settings.simulation_defaults.duration,
    timestep: settings.simulation_defaults.timestep,
  });
}

export function useWebSettingsBootstrap(): void {
  useEffect(() => {
    const cached = readCachedSettings();
    if (cached) {
      applyFontScale(cached.appearance.font_scale);
    }

    let cancelled = false;
    fetchSettings()
      .then((settings) => {
        if (cancelled) return;
        applySettingsLocally(settings);
        hydrateSimulationDefaults(settings);
      })
      .catch(() => {
        // Offline / backend not up yet: fall back to the cache if present.
        if (cancelled || !cached) return;
        hydrateSimulationDefaults(cached);
      });

    return () => {
      cancelled = true;
    };
  }, []);
}
