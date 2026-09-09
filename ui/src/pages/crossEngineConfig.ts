/**
 * Perturbation-config contract for the Cross-Engine Robustness Dashboard.
 *
 * Lives beside `CrossEngineDashboard.tsx` rather than inside it so the page
 * module only exports its component (Fast Refresh constraint) and so the
 * validation rules are unit-testable without rendering.
 *
 * The bounds mirror `CrossEnginePerturbationConfig` in
 * `src/api/routes/cross_engine.py`; keeping them in sync is what stops an
 * invalid config from reaching the API (issue #8891).
 */

export interface PerturbationConfig {
  t_end: number;
  dt: number;
  noise_amplitude: number;
  n_trials: number;
  seed: number;
}

export type ConfigKey = keyof PerturbationConfig;

export interface ConfigField {
  key: ConfigKey;
  label: string;
  min: number;
  step: number;
  /** Whole-number fields reject fractional input. */
  integer: boolean;
}

export const CONFIG_FIELDS: readonly ConfigField[] = [
  { key: 't_end', label: 'Duration (s)', min: 0.1, step: 0.1, integer: false },
  { key: 'dt', label: 'Timestep (s)', min: 0.001, step: 0.001, integer: false },
  { key: 'noise_amplitude', label: 'Noise amplitude', min: 0, step: 0.01, integer: false },
  { key: 'n_trials', label: 'Trials', min: 1, step: 1, integer: true },
  { key: 'seed', label: 'Seed', min: 0, step: 1, integer: true },
];

const DEFAULT_CONFIG: PerturbationConfig = {
  t_end: 1.0,
  dt: 0.01,
  noise_amplitude: 0.05,
  n_trials: 10,
  seed: 42,
};

/** Raw, as-typed text for every config field. */
export type ConfigText = Record<ConfigKey, string>;

export const DEFAULT_CONFIG_TEXT: ConfigText = {
  t_end: String(DEFAULT_CONFIG.t_end),
  dt: String(DEFAULT_CONFIG.dt),
  noise_amplitude: String(DEFAULT_CONFIG.noise_amplitude),
  n_trials: String(DEFAULT_CONFIG.n_trials),
  seed: String(DEFAULT_CONFIG.seed),
};

/** Poll cadence for the study status endpoint. */
export const POLL_INTERVAL_MS = 1000;

/**
 * Hard cap on how long the status endpoint is polled.
 *
 * Without it, a backend restart or a dropped task leaves the page polling
 * every second forever with no way out but a reload.
 */
export const POLL_DEADLINE_MS = 10 * 60 * 1000;

function validateField(field: ConfigField, raw: string): string | null {
  const trimmed = raw.trim();
  if (trimmed === '') {
    return `${field.label} is required.`;
  }
  const value = Number(trimmed);
  if (!Number.isFinite(value)) {
    return `${field.label} must be a number.`;
  }
  if (field.integer && !Number.isInteger(value)) {
    return `${field.label} must be a whole number.`;
  }
  if (value < field.min) {
    return `${field.label} must be at least ${field.min}.`;
  }
  return null;
}

/**
 * Validate raw config text.
 *
 * Postcondition: the returned record is empty exactly when every field
 * parses to a number satisfying its own bound and `dt < t_end` — the
 * cross-field constraint the API enforces.
 */
export function validateConfigText(text: ConfigText): Partial<Record<ConfigKey, string>> {
  const errors: Partial<Record<ConfigKey, string>> = {};
  for (const field of CONFIG_FIELDS) {
    const message = validateField(field, text[field.key]);
    if (message !== null) {
      errors[field.key] = message;
    }
  }
  if (errors.dt === undefined && errors.t_end === undefined) {
    if (Number(text.dt) >= Number(text.t_end)) {
      errors.dt = 'Timestep (s) must be smaller than the duration.';
    }
  }
  return errors;
}

/** Parse validated config text into the numeric request payload. */
export function parseConfigText(text: ConfigText): PerturbationConfig {
  return {
    t_end: Number(text.t_end),
    dt: Number(text.dt),
    noise_amplitude: Number(text.noise_amplitude),
    n_trials: Number(text.n_trials),
    seed: Number(text.seed),
  };
}
