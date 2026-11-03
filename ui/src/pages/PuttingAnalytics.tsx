/**
 * Putting Analytics page (#14235): the web twin of the desktop tool.
 */
import { useEffect, useState } from 'react';
import { apiFetch, describeApiError } from '@/api/fetch';
import {
  DEFAULT_FORM, MODE_LABELS, PUTTING_ANALYTICS_PATH, STUDY_MODES,
  resultRows, runStudy, studyArgs, type Row, type StudyForm, type StudyMode,
} from '@/api/puttingAnalytics';

const NUMBER_FIELDS: [keyof StudyForm, string, number][] = [
  ['distanceFt', 'Distance (ft)', 0.1], ['gradePct', 'Grade (%)', 0.1],
  ['aspectDeg', 'Aspect (deg)', 1], ['stimp', 'Stimp', 0.1],
  ['lineSdDeg', 'Start-line SD (deg)', 0.1], ['speedSd', 'Speed SD (fraction)', 0.01],
  ['n', 'Rollouts per action', 1], ['seed', 'Seed', 1],
];
const TEXT_FIELDS: [keyof StudyForm, string, string][] = [
  ['baselineDistances', 'Baseline distances (ft)', 'your distances: 2 5 10 20 40'],
  ['baselinePutts', 'Baseline putts', 'your expected putts at each distance'],
  ['aims', 'Aims (deg)', ''], ['paces', 'Paces past hole (ft)', ''],
];
const EXTRA_FIELDS: Partial<Record<StudyMode, [keyof StudyForm, string]>> = {
  misread_cost: ['reads', 'Reads'], library_study: ['classes', 'Classes'],
};
const INPUT = 'w-full rounded bg-surface border border-token-border px-2 py-1 text-sm text-token-text';
const LABEL = 'flex flex-col gap-1 text-xs text-token-muted';

export function PuttingAnalyticsPage() {
  const [form, setForm] = useState<StudyForm>(DEFAULT_FORM);
  const [mode, setMode] = useState<StudyMode>('optimize');
  const [objectives, setObjectives] = useState<string[]>([DEFAULT_FORM.objective]);
  const [rows, setRows] = useState<Row[]>([]);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState('Enter your own expected-putts curve; no reference data ships.');

  useEffect(() => {
    apiFetch<{ objectives: string[] }>(`${PUTTING_ANALYTICS_PATH}/limits`)
      .then((limits) => setObjectives(limits.objectives)).catch(() => undefined);
  }, []);

  const extraField = EXTRA_FIELDS[mode];
  const set = (key: keyof StudyForm, value: string | number) =>
    setForm((curr) => ({ ...curr, [key]: value }));

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
            <select className={INPUT} value={mode} onChange={(e) => setMode(e.target.value as StudyMode)}>
              {STUDY_MODES.map((m) => (<option key={m} value={m}>{MODE_LABELS[m]}</option>))}
            </select>
          </label>
          <div className="grid grid-cols-2 gap-3">
            {NUMBER_FIELDS.map(([k, label, step]) => (
              <label key={k} className={LABEL}>
                {label}
                <input type="number" step={step} className={INPUT} value={form[k]} onChange={(e) => set(k, Number(e.target.value))} />
              </label>
            ))}
          </div>
          <label className={LABEL}>
            Objective
            <select className={INPUT} value={form.objective} onChange={(e) => set('objective', e.target.value)}>
              {objectives.map((o) => (<option key={o}>{o}</option>))}
            </select>
          </label>
          {TEXT_FIELDS.map(([k, label, placeholder]) => (
            <label key={k} className={LABEL}>
              {label}
              <input className={INPUT} placeholder={placeholder} value={form[k]} onChange={(e) => set(k, e.target.value)} />
            </label>
          ))}
          {extraField && (
            <label className={LABEL}>
              {extraField[1]}
              <textarea rows={4} className={`${INPUT} font-mono`} value={form[extraField[0]]} onChange={(e) => set(extraField[0], e.target.value)} />
            </label>
          )}
          <button type="button" disabled={running} onClick={run} className="rounded-lg bg-primary px-3 py-2 text-sm font-medium text-white hover:opacity-90 disabled:opacity-50">
            Run study
          </button>
        </section>
        <section aria-label="Study results" className="flex min-w-0 flex-1 flex-col gap-3">
          <p role="status" className="text-sm text-token-muted">{status}</p>
          <div className="overflow-x-auto">
            <table className="w-full border border-token-border text-sm">
              <thead>
                <tr className="bg-surface-raised text-token-muted">
                  <th className="px-2 py-1 text-left">Quantity</th>
                  <th className="px-2 py-1 text-left">Value</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([name, value], index) => (
                  <tr key={`${name}-${index}`} className="border-t border-token-border text-token-text">
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
