/**
 * Request controls of the Variability & Consistency section (NSX-100, #12729).
 * Every list is the backend catalogue's; the controls only collect choices and
 * report them. An engine that cannot supply derivatives is listed with its
 * UNAVAILABLE reason. Nothing is computed.
 */

import { useEffect, useState } from 'react';
import type { SensitivityCatalogue } from '@/api/loopNullSpace';
import { parseChannels, type VariabilityChoices } from './variabilityRequest';

const FIELD = 'bg-gray-900 border border-gray-600 rounded px-2 py-1 text-sm text-gray-100';
const DEFAULT_ENGINE = 'pinocchio';

export interface ControlsState {
  enabled: boolean;
  engine: string;
  error: string | null;
  choices: VariabilityChoices;
}

interface Props {
  catalogue: SensitivityCatalogue;
  onChange: (state: ControlsState) => void;
}

const toggle = (list: string[], id: string, on: boolean) =>
  on ? [...list, id] : list.filter((x) => x !== id);

export function VariabilityControls({ catalogue, onChange }: Props) {
  const [enabled, setEnabled] = useState(false);
  const [name, setName] = useState('disturbance');
  const [stage, setStage] = useState(catalogue.stages[0]?.id ?? '');
  const [spectrum, setSpectrum] = useState(catalogue.spectra[0]?.id ?? '');
  const [persistent, setPersistent] = useState(false);
  const [channels, setChannels] = useState('');
  const [units, setUnits] = useState('N m');
  const [bandwidth, setBandwidth] = useState('');
  const [mechanisms, setMechanisms] = useState<string[]>([]);
  const [strategies, setStrategies] = useState<string[]>(catalogue.strategies.map((s) => s.id));
  const [outcome, setOutcome] = useState(
    catalogue.default_outcome_priority[0] ?? catalogue.outcomes[0]?.id ?? '',
  );
  const [engine, setEngine] = useState(
    catalogue.engines.find((e) => e.id === DEFAULT_ENGINE)?.id ?? catalogue.engines[0]?.id ?? '',
  );

  useEffect(() => {
    let error: string | null = null;
    let parsed: ReturnType<typeof parseChannels> = [];
    const band = bandwidth.trim() === '' ? null : Number(bandwidth);
    try {
      parsed = parseChannels(channels);
      if (band !== null && !Number.isFinite(band)) throw new Error('bandwidth is not a number');
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
    const every = strategies.length === catalogue.strategies.length;
    onChange({
      enabled,
      engine,
      error,
      choices: {
        disturbances: error
          ? []
          : [{ name, stage, spectrum, persistent, channels: parsed, units, bandwidthHz: band }],
        mechanisms,
        strategies: every ? null : strategies,
        outcomePriority: outcome ? [outcome] : [],
      },
    });
  }, [
    enabled, name, stage, spectrum, persistent, channels, units, bandwidth,
    mechanisms, strategies, outcome, engine, catalogue, onChange,
  ]); // prettier-ignore

  const select = (
    label: string,
    value: string,
    set: (v: string) => void,
    options: Array<{ id: string; text: string }>,
  ) => (
    <label className="text-xs text-gray-300 flex flex-col gap-1">
      {label}
      <select className={FIELD} value={value} onChange={(e) => set(e.target.value)}>
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {o.text}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <fieldset
      data-testid="nsx-variability-controls"
      className="border border-gray-700 rounded p-2 mb-3"
    >
      <legend className="text-xs text-gray-200 px-1">Variability and Consistency Options</legend>
      <label className="text-xs text-gray-300 flex gap-2 items-center mb-2">
        <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
        Run variability analysis
      </label>
      <div className="flex flex-wrap gap-3 items-end mb-2">
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Disturbance
          <input className={FIELD} value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        {select(
          'Stage',
          stage,
          setStage,
          catalogue.stages.map((s) => ({ id: s.id, text: s.label })),
        )}
        {select(
          'Spectrum',
          spectrum,
          setSpectrum,
          catalogue.spectra.map((s) => ({ id: s.id, text: s.label })),
        )}
        <label className="text-xs text-gray-300 flex gap-2 items-center">
          <input
            type="checkbox"
            checked={persistent}
            onChange={(e) => setPersistent(e.target.checked)}
          />
          Persistent per-trial state
        </label>
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Channels and amplitudes (variance)
          <input
            className={FIELD}
            placeholder="channel=variance, channel=variance"
            value={channels}
            onChange={(e) => setChannels(e.target.value)}
          />
        </label>
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Units
          <input className={FIELD} value={units} onChange={(e) => setUnits(e.target.value)} />
        </label>
        <label className="text-xs text-gray-300 flex flex-col gap-1">
          Bandwidth (Hz)
          <input
            className={FIELD}
            value={bandwidth}
            onChange={(e) => setBandwidth(e.target.value)}
          />
        </label>
      </div>
      <div className="flex flex-wrap gap-4 mb-2">
        <div>
          <p className="text-xs text-gray-200">Mechanisms</p>
          {catalogue.mechanisms.map((m) => (
            <label key={m.id} className="text-xs text-gray-300 flex gap-2 items-center">
              <input
                type="checkbox"
                checked={mechanisms.includes(m.id)}
                onChange={(e) => setMechanisms(toggle(mechanisms, m.id, e.target.checked))}
              />
              {m.id}: {m.label}
              {m.note ? ` (${m.note})` : ''}
            </label>
          ))}
        </div>
        <div>
          <p className="text-xs text-gray-200">Strategies</p>
          {catalogue.strategies.map((s) => (
            <label key={s.id} className="text-xs text-gray-300 flex gap-2 items-center">
              <input
                type="checkbox"
                checked={strategies.includes(s.id)}
                onChange={(e) => setStrategies(toggle(strategies, s.id, e.target.checked))}
              />
              {s.id}: {s.name}
              {s.availability !== 'implemented' ? ` (${s.availability}: ${s.reason})` : ''}
            </label>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-3">
        {select(
          'Primary outcome',
          outcome,
          setOutcome,
          catalogue.outcomes.map((o) => ({ id: o.id, text: o.label })),
        )}
        {select(
          'Engine',
          engine,
          setEngine,
          catalogue.engines.map((e) => ({
            id: e.id,
            text: e.available ? e.id : `${e.id} (UNAVAILABLE: ${e.reason})`,
          })),
        )}
      </div>
    </fieldset>
  );
}
