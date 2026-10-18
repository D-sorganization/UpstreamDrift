/**
 * Builds the sensitivity request options from the user's choices (NSX-100,
 * #12729). Mirrors `src/tools/grip_wrench_plots/variability_request.py`: every
 * id must be in the backend catalogue and the amplitude is the diagonal
 * covariance entry the user typed, placed as typed (nothing is derived).
 */

import type { SensitivityCatalogue, SensitivityOptionsPayload } from '@/api/loopNullSpace';

export type ChannelAmplitude = [name: string, variance: number];

export interface DisturbanceChoice {
  name: string;
  stage: string;
  spectrum: string;
  persistent: boolean;
  channels: ChannelAmplitude[];
  units: string;
  bandwidthHz: number | null;
}

export interface VariabilityChoices {
  disturbances: DisturbanceChoice[];
  mechanisms: string[];
  /** `null` is the whole strategy catalogue. */
  strategies: string[] | null;
  outcomePriority: string[];
}

/** `"shoulder=0.004, elbow=0.009"` as pairs; throws `Error` on anything else. */
export function parseChannels(text: string): ChannelAmplitude[] {
  const pairs: ChannelAmplitude[] = text.split(',').map((raw) => {
    const part = raw.trim();
    const at = part.indexOf('=');
    const name = at < 0 ? '' : part.slice(0, at).trim();
    if (!name) throw new Error(`expected name=variance, got "${part}"`);
    const value = Number(part.slice(at + 1));
    if (part.slice(at + 1).trim() === '' || !Number.isFinite(value) || value < 0) {
      throw new Error(`variance of "${name}" must be a finite non-negative number`);
    }
    return [name, value];
  });
  if (new Set(pairs.map((p) => p[0])).size !== pairs.length) {
    throw new Error('name each channel once');
  }
  return pairs;
}

const ids = (entries: Array<{ id: string }>) => new Set(entries.map((e) => e.id));

function check(values: string[], allowed: Set<string>, what: string) {
  const unknown = values.filter((v) => !allowed.has(v));
  if (unknown.length > 0) {
    throw new Error(`${what} ${JSON.stringify(unknown)} not in the backend catalogue`);
  }
}

/** The `SensitivityOptions` JSON for `choices`; throws when an id is not catalogued. */
export function buildSensitivityOptions(
  choices: VariabilityChoices,
  catalogue: SensitivityCatalogue,
): SensitivityOptionsPayload {
  if (choices.disturbances.length === 0) throw new Error('declare at least one disturbance');
  check(choices.mechanisms, ids(catalogue.mechanisms), 'mechanism');
  if (choices.strategies) check(choices.strategies, ids(catalogue.strategies), 'strategy');
  if (choices.outcomePriority.length === 0) throw new Error('name at least one outcome');
  check(choices.outcomePriority, ids(catalogue.outcomes), 'outcome');
  return {
    disturbance_spec: choices.disturbances.map((d) => {
      check([d.stage], ids(catalogue.stages), 'stage');
      check([d.spectrum], ids(catalogue.spectra), 'spectrum');
      if (d.channels.length === 0 || !d.name.trim() || !d.units.trim()) {
        throw new Error('a disturbance needs a name, units and a channel');
      }
      return {
        name: d.name.trim(),
        stage: d.stage,
        channels: d.channels.map(([n]) => n),
        units: d.units.trim(),
        covariance: d.channels.map((_, i) => d.channels.map(([, v], j) => (i === j ? v : 0))),
        spectrum: d.persistent ? null : d.spectrum,
        persistent: d.persistent,
        bandwidth_hz: d.bandwidthHz,
        band_hz: null,
        signal_gain: null,
        signal_bounds: null,
      };
    }),
    mechanisms: choices.mechanisms,
    strategy_set: choices.strategies,
    outcome_priority: choices.outcomePriority,
  };
}
