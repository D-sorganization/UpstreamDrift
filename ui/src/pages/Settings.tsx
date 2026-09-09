/**
 * Settings Page — issue #7457 (parity epic #7462).
 *
 * Web counterpart of the desktop settings dialog (deliberate subset):
 *   - Appearance: theme selection (server theme list) + font scale
 *   - Notifications: toast duration + verbosity
 *   - Simulation defaults: default engine, duration, timestep
 *   - Diagnostics: opens the diagnostics panel
 *
 * Settings load from the server on mount (`GET /api/v1/settings`) and are
 * persisted with an explicit Save button (`PUT /api/v1/settings`). Saved
 * settings apply immediately: theme tokens are re-fetched and applied as
 * CSS variables, the font scale is set as a root CSS variable, and the
 * localStorage cache (cache only) is refreshed.
 *
 * Unsaved edits are tracked against a pristine snapshot (#8892). Leaving the
 * page -- the back arrow, a tab close, a reload -- confirms first, and the
 * Save button is disabled and relabelled according to that dirty state, so a
 * route change or tab close can never discard edits silently.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowLeft, Stethoscope } from 'lucide-react';
import {
  DEFAULT_WEB_SETTINGS,
  applySettingsLocally,
  fetchSettings,
  fetchThemeList,
  readCachedSettings,
  saveSettings,
  setActiveTheme,
  type NotificationVerbosity,
  type WebSettings,
} from '@/api/settingsClient';
import { applyThemeToCSSVariables, fetchActiveTheme } from '@/api/themeClient';
import { useToast } from '@/components/ui/Toast';
import { useUIStore, useEngineStore, useSimulationStore } from '@/stores';
import { useUnsavedChangesGuard } from '@/hooks/useUnsavedChangesGuard';

type LoadState = 'loading' | 'loaded' | 'error';

const VERBOSITY_OPTIONS: { value: NotificationVerbosity; label: string }[] = [
  { value: 'all', label: 'All notifications' },
  { value: 'errors', label: 'Errors and warnings only' },
  { value: 'silent', label: 'Silent' },
];

function SectionCard({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-gray-800 rounded-lg border border-gray-700 p-4">
      <h2 className="text-sm font-semibold text-gray-200">{title}</h2>
      <p className="text-xs text-gray-400 mt-1 mb-4">{description}</p>
      <div className="space-y-3">{children}</div>
    </section>
  );
}

function FieldRow({ label, htmlFor, children }: { label: string; htmlFor: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <label htmlFor={htmlFor} className="text-sm text-gray-300">
        {label}
      </label>
      {children}
    </div>
  );
}

const INPUT_CLASS =
  'bg-gray-700 border border-gray-600 text-gray-100 text-sm rounded px-2 py-1.5 w-48 focus:outline-none focus:ring-2 focus:ring-blue-500';

export function SettingsPage() {
  const { showSuccess, showError } = useToast();
  const setDiagnosticsOpen = useUIStore((s) => s.setDiagnosticsOpen);
  const engines = useEngineStore((s) => s.engines);
  const hydrateDefaults = useSimulationStore((s) => s.hydrateDefaults);

  const [settings, setSettings] = useState<WebSettings>(
    () => readCachedSettings() ?? DEFAULT_WEB_SETTINGS,
  );
  const [themeNames, setThemeNames] = useState<string[]>([]);
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [saving, setSaving] = useState(false);

  // Dirty tracking (#8892). `pristine` is the last state that is known to
  // match the server: whatever was loaded on mount, or whatever a successful
  // save returned. Comparison is a deep value compare via JSON -- WebSettings
  // is a small, flat, JSON-round-trippable object, so this is exact rather
  // than a heuristic, and it correctly reports "clean" when the user edits a
  // field and then puts it back.
  const [pristine, setPristine] = useState<WebSettings>(settings);
  const isDirty = useMemo(
    () => JSON.stringify(settings) !== JSON.stringify(pristine),
    [settings, pristine],
  );
  const { guardedNavigate } = useUnsavedChangesGuard(isDirty);

  // Load settings + theme list on mount (localStorage is only a cache; the
  // server file is authoritative).
  useEffect(() => {
    let cancelled = false;
    fetchSettings()
      .then((loaded) => {
        if (cancelled) return;
        setSettings(loaded);
        setPristine(loaded);
        setLoadState('loaded');
      })
      .catch(() => {
        if (cancelled) return;
        setLoadState('error');
      });
    fetchThemeList()
      .then((list) => {
        if (cancelled) return;
        setThemeNames(Object.keys(list.themes));
      })
      .catch(() => {
        // Theme list is optional; the select degrades to the current value.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const update = useCallback(<K extends keyof WebSettings>(section: K, patch: Partial<WebSettings[K]>) => {
    setSettings((prev) => ({ ...prev, [section]: { ...prev[section], ...patch } }));
  }, []);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const saved = await saveSettings(settings);
      // Apply immediately: theme round-trips through the theme manager so
      // the desktop and web stay in sync, tokens are re-fetched and applied.
      try {
        await setActiveTheme(saved.appearance.theme_id);
        const active = await fetchActiveTheme();
        applyThemeToCSSVariables(active.colors);
        document.documentElement.dataset.theme = active.name;
      } catch {
        showError('Settings saved, but the theme could not be applied');
      }
      applySettingsLocally(saved);
      hydrateDefaults({
        duration: saved.simulation_defaults.duration,
        timestep: saved.simulation_defaults.timestep,
      });
      setSettings(saved);
      setPristine(saved);
      showSuccess('Settings saved');
    } catch (err) {
      showError(err instanceof Error ? err.message : 'Failed to save settings');
    } finally {
      setSaving(false);
    }
  }, [settings, showSuccess, showError, hydrateDefaults]);

  const themeOptions = themeNames.includes(settings.appearance.theme_id)
    ? themeNames
    : [settings.appearance.theme_id, ...themeNames];

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center gap-4">
        <button
          type="button"
          onClick={() => guardedNavigate('/')}
          className="p-1.5 rounded hover:bg-gray-700 transition-colors"
          aria-label="Back to dashboard"
        >
          <ArrowLeft className="w-5 h-5" aria-hidden="true" />
        </button>
        <div>
          <h1 className="text-lg font-semibold text-gray-100">Settings</h1>
          <p className="text-xs text-gray-400 mt-0.5">
            Preferences persist server-side and apply to both browser and Tauri modes
          </p>
        </div>
      </div>

      {loadState === 'error' && (
        <div className="bg-red-900/30 border-b border-red-800 px-6 py-3 text-sm text-red-300">
          Could not load settings from the server — showing cached values. Saving will retry.
        </div>
      )}

      <div className="p-6 max-w-2xl mx-auto space-y-6">
        {/* Appearance */}
        <SectionCard
          title="Appearance"
          description="Theme and text size. Theme changes apply without reload."
        >
          <FieldRow label="Theme" htmlFor="settings-theme">
            <select
              id="settings-theme"
              className={INPUT_CLASS}
              value={settings.appearance.theme_id}
              onChange={(e) => update('appearance', { theme_id: e.target.value })}
            >
              {themeOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </FieldRow>
          <FieldRow label={`Font scale (${settings.appearance.font_scale.toFixed(2)}×)`} htmlFor="settings-font-scale">
            <input
              id="settings-font-scale"
              type="range"
              min={0.5}
              max={2.0}
              step={0.05}
              className="w-48 accent-blue-500"
              value={settings.appearance.font_scale}
              onChange={(e) => update('appearance', { font_scale: Number(e.target.value) })}
            />
          </FieldRow>
        </SectionCard>

        {/* Notifications */}
        <SectionCard
          title="Notifications"
          description="How long toasts stay visible and which ones are shown."
        >
          <FieldRow label="Toast duration (ms)" htmlFor="settings-toast-duration">
            <input
              id="settings-toast-duration"
              type="number"
              min={500}
              max={60000}
              step={500}
              className={INPUT_CLASS}
              value={settings.notifications.toast_duration_ms}
              onChange={(e) =>
                update('notifications', { toast_duration_ms: Number(e.target.value) })
              }
            />
          </FieldRow>
          <FieldRow label="Verbosity" htmlFor="settings-verbosity">
            <select
              id="settings-verbosity"
              className={INPUT_CLASS}
              value={settings.notifications.verbosity}
              onChange={(e) =>
                update('notifications', {
                  verbosity: e.target.value as NotificationVerbosity,
                })
              }
            >
              {VERBOSITY_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </FieldRow>
        </SectionCard>

        {/* Simulation defaults */}
        <SectionCard
          title="Simulation defaults"
          description="Defaults applied when the app starts. In-session changes on the simulation page are never overwritten."
        >
          <FieldRow label="Default engine" htmlFor="settings-default-engine">
            <select
              id="settings-default-engine"
              className={INPUT_CLASS}
              value={settings.simulation_defaults.default_engine}
              onChange={(e) =>
                update('simulation_defaults', { default_engine: e.target.value })
              }
            >
              {engines.map((engine) => (
                <option key={engine.name} value={engine.name}>
                  {engine.displayName}
                </option>
              ))}
            </select>
          </FieldRow>
          <FieldRow label="Duration (s)" htmlFor="settings-duration">
            <input
              id="settings-duration"
              type="number"
              min={0.1}
              max={300}
              step={0.1}
              className={INPUT_CLASS}
              value={settings.simulation_defaults.duration}
              onChange={(e) =>
                update('simulation_defaults', { duration: Number(e.target.value) })
              }
            />
          </FieldRow>
          <FieldRow label="Timestep (s)" htmlFor="settings-timestep">
            <input
              id="settings-timestep"
              type="number"
              min={0.0001}
              max={1}
              step={0.0001}
              className={INPUT_CLASS}
              value={settings.simulation_defaults.timestep}
              onChange={(e) =>
                update('simulation_defaults', { timestep: Number(e.target.value) })
              }
            />
          </FieldRow>
        </SectionCard>

        {/* Diagnostics */}
        <SectionCard
          title="Diagnostics"
          description="Inspect backend health, engine availability, and integration status."
        >
          <button
            type="button"
            onClick={() => setDiagnosticsOpen(true)}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 text-sm rounded transition-colors"
          >
            <Stethoscope className="w-4 h-4" aria-hidden="true" />
            Open diagnostics panel
          </button>
        </SectionCard>

        {/* Save */}
        <div className="flex items-center justify-end gap-3">
          <p
            className="text-xs text-gray-400"
            data-testid="settings-dirty-state"
            aria-live="polite"
          >
            {isDirty ? 'You have unsaved changes.' : 'All changes saved.'}
          </p>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving || loadState === 'loading' || !isDirty}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors"
          >
            {saving ? 'Saving…' : isDirty ? 'Save changes •' : 'Save settings'}
          </button>
        </div>
      </div>
    </div>
  );
}
