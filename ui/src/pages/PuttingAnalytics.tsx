/**
 * Putting Analytics page (#14235): the web twin of the desktop tool.
 *
 * Collects the same inputs as `src/tools/putting_analytics_gui/gui.py`, runs
 * the selected study through the `/tools/putting-analytics` API (the shared
 * putting service the desktop tool calls), and shows the answer as the same
 * Quantity/Value rows. The page computes nothing itself.
 */

import { useEffect, useState } from 'react';
import { apiFetch, describeApiError } from '@/api/fetch';
import {
  DEFAULT_FORM,
  MODE_LABELS,
  PUTTING_ANALYTICS_PATH,
  STUDY_MODES,
  resultRows,
  runStudy,
  studyArgs,
  type Row,
  type StudyForm,
  type StudyMode,
} from '@/api/puttingAnalytics';

const NUMBER_FIELDS: [keyof StudyForm, string, number][] = [
  ['distanceFt', 'Distance (ft)', 0.1],
  ['gradePct', 'Grade (%)', 0.1],
  ['aspectDeg', 'Aspect (deg)', 1],
  ['stimp', 'Stimp', 0.1],
  ['lineSdDeg', 'Start-line SD (deg)', 0.1],
  ['speedSd', 'Speed SD (fraction)', 0.01],
  ['n', 'Rollouts per action', 1],
  ['seed', 'Seed', 1],
];
const TEXT_FIELDS: [keyof StudyForm, string, string][] = [
  ['baselineDistances', 'Baseline distances (ft)', 'your distances: 2 5 10 20 40'],
  ['baselinePutts', 'Baseline putts', 'your expected putts at each distance'],
  ['aims', 'Aims (deg)', ''],
  ['paces', 'Paces past hole (ft)', ''],
];

/** The study-specific list each mode adds, as on the desktop form. */
const EXTRA_FIELDS: Partial<Record<StudyMode, [keyof StudyForm, string]>> = {
  misread_cost: ['reads', 'Reads'],
  library_study: ['classes', 'Classes'],
};

const INPUT = 'w-full rounded bg-gray-900 border border-gray-700 px-2 py-1 text-sm text-white';
const LABEL = 'flex flex-col gap-1 text-xs text-gray-300';

export function PuttingAnalyticsPage() {
  const [form, setForm] = useState<StudyForm>(DEFAULT_FORM);
  const [mode, setMode] = useState<StudyMode>('optimize');
  const [objectives, setObjectives] = useState<string[]>([DEFAULT_FORM.objective]);
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState(
    'Enter your own expected-putts curve; no reference data ships.',
  );

  useEffect(() => {
    apiFetch<{ objectives: string[] }>(`${PUTTING_ANALYTICS_PATH}/limits`)
      .then((limits) => setObjectives(limits.objectives))
      .catch(() => undefined);
  }, []);

  const extraField = EXTRA_FIELDS[mode];
  const set = (key: keyof StudyForm, value: string | number) =>
    setForm((current) => ({ ...current, [key]: value }));

  async function run() {
    let args;
    try {
      args = studyArgs(form, mode);
    } catch (err) {
      setStatus(`Check the inputs: ${(err as Error).message}`);
      return;
    }
    setRunning(true);
    setStatus(`Running: ${MODE_LABELS[mode]}...`);
    try {
      setRows(resultRows(mode, await runStudy(mode, args)));
      setStatus(`Done: ${MODE_LABELS[mode]}.`);
    } catch (err) {
      setStatus(`The study failed: ${describeApiError(err, 'Putting Analytics', 'no answer')}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="flex flex-col gap-4 p-4 sm:p-6">
      <h1 className="text-2xl font-bold text-white">Putting Analytics</h1>
      <div className="flex flex-col gap-4 lg:flex-row">
        <section aria-label="Study inputs" className="flex flex-col gap-3 lg:w-96">
          <label className={LABEL}>
            Study
            <select
              className={INPUT}
              value={mode}
              onChange={(e) => setMode(e.target.value as StudyMode)}
            >
              {STUDY_MODES.map((m) => (
                <option key={m} value={m}>
                  {MODE_LABELS[m]}
                </option>
              ))}
            </select>
          </label>
          <div className="grid grid-cols-2 gap-3">
            {NUMBER_FIELDS.map(([key, label, step]) => (
              <label key={key} className={LABEL}>
                {label}
                <input
                  type="number"
                  step={step}
                  className={INPUT}
                  value={form[key]}
                  onChange={(e) => set(key, Number(e.target.value))}
                />
              </label>
            ))}
          </div>
          <label className={LABEL}>
            Objective
            <select
              className={INPUT}
              value={form.objective}
              onChange={(e) => set('objective', e.target.value)}
            >
              {objectives.map((o) => (
                <option key={o}>{o}</option>
              ))}
            </select>
          </label>
          {TEXT_FIELDS.map(([key, label, placeholder]) => (
            <label key={key} className={LABEL}>
              {label}
              <input
                className={INPUT}
                placeholder={placeholder}
                value={form[key]}
                onChange={(e) => set(key, e.target.value)}
              />
            </label>
          ))}
          {extraField && (
            <label className={LABEL}>
              {extraField[1]}
              <textarea
                rows={4}
                className={`${INPUT} font-mono`}
                value={form[extraField[0]]}
                onChange={(e) => set(extraField[0], e.target.value)}
              />
            </label>
          )}
          <button
            type="button"
            disabled={running}
            onClick={run}
            className="rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-500 disabled:opacity-50"
          >
            Run study
          </button>
        </section>
        <section aria-label="Study results" className="flex min-w-0 flex-1 flex-col gap-3">
          <p role="status" className="text-sm text-gray-300">
            {status}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full border border-gray-700 text-sm">
              <thead>
                <tr className="bg-gray-800 text-gray-300">
                  <th className="px-2 py-1 text-left">Quantity</th>
                  <th className="px-2 py-1 text-left">Value</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([name, value], index) => (
                  <tr key={`${name}-${index}`} className="border-t border-gray-700 text-gray-200">
                    <td className="px-2 py-1 font-mono">{name}</td>
                    <td className="px-2 py-1">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
