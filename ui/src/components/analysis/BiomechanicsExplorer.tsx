/** Scientific values and conversions are supplied by the shared Python service. */
import { useRef, useState } from 'react';
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { apiFetch } from '@/api/fetch';

interface Channel { values: (number | null)[]; unit: string; definition: string; frame: string }
interface Result {
  times: number[]; channels: Record<string, Channel>; source: string;
  unavailable: Record<string, string>; events?: Record<string, number>;
  summaries?: Record<string, number>;
}
const COLORS = ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa'];

function download(content: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  URL.revokeObjectURL(url);
}

export function BiomechanicsExplorer() {
  const [raw, setRaw] = useState<unknown>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [colors, setColors] = useState<Record<string, string>>({});
  const [unit, setUnit] = useState('deg');
  const [min, setMin] = useState('');
  const [max, setMax] = useState('');
  const [showEvents, setShowEvents] = useState(true);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const plots = useRef<HTMLDivElement>(null);
  const validRange = (min === '' || Number.isFinite(Number(min))) &&
    (max === '' || Number.isFinite(Number(max))) &&
    (min === '' || max === '' || Number(min) < Number(max));

  async function display(value: unknown, angleUnit = unit, preserveSelection = false) {
    const prepared = await apiFetch<Result>('/api/biomechanics/display', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ result: value, angle_unit: angleUnit }),
    });
    setRaw(value); setResult(prepared);
    if (!preserveSelection) setSelected(Object.keys(prepared.channels).slice(0, 4));
  }
  async function run(operation: () => Promise<void>) {
    setBusy(true); setError('');
    try { await operation(); }
    catch (e) { setError(e instanceof Error ? e.message : 'Unable to Load Biomechanics'); }
    finally { setBusy(false); }
  }
  const units = [...new Set(selected.map((key) => result?.channels[key]?.unit).filter(Boolean))];
  const rows = result?.times.map((time, index) => ({
    time, ...Object.fromEntries(selected.map((key) => [key, result.channels[key].values[index]])),
  })) ?? [];

  return <section className="space-y-3 p-3" aria-label="Biomechanics Explorer">
    <h3 className="font-semibold">Biomechanics Explorer</h3>
    <p className="text-sm text-gray-400">Compare joint conventions, segment motion, golf metrics and centers of mass. Missing samples remain gaps. Each unit has its own plot.</p>
    <div className="flex flex-wrap gap-3">
      <label>Import Model Binding<input aria-label="Import Model Binding JSON" type="file" accept=".json,application/json" disabled={busy} onChange={(event) => {
        const file = event.target.files?.[0];
        if (file) void run(async () => {
          const binding = JSON.parse(await file.text());
          await apiFetch('/api/biomechanics/bindings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(binding) });
          setNotice('Model Binding Loaded. Start a New Simulation, Then Load Recorded Biomechanics.');
        });
      }} /></label>
      <button disabled={busy} onClick={() => void run(async () => display(await apiFetch('/api/biomechanics/results')))}>Load Recorded Biomechanics</button>
      <label>Import Trajectory or Result JSON<input aria-label="Import Biomechanics JSON" type="file" accept=".json,application/json" disabled={busy} onChange={(event) => {
        const file = event.target.files?.[0];
        if (file) void run(async () => {
          const payload = JSON.parse(await file.text());
          const computed = payload.channels ? payload : await apiFetch('/api/biomechanics/compute', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
          });
          await display(computed);
        });
      }} /></label>
      <label>Angle Units<select aria-label="Angle Units" value={unit} disabled={busy} onChange={(event) => {
        const next = event.target.value; setUnit(next);
        if (raw) void run(() => display(raw, next, true));
      }}><option value="deg">Degrees</option><option value="rad">Radians</option></select></label>
    </div>
    {busy && <p role="status">Loading Biomechanics…</p>}
    {notice && <p role="status">{notice}</p>}
    {error && <p role="alert">{error}</p>}
    {result && <>
      <p className="text-xs">Source: {result.source}</p>
      <details><summary>Availability and Definitions</summary>
        {Object.entries(result.unavailable ?? {}).map(([key, reason]) => <p key={key}>{key}: {reason}</p>)}
        {Object.entries(result.channels).map(([key, channel]) => <p key={key}>{key}: {channel.definition} · Frame: {channel.frame} · Unit: {channel.unit}</p>)}
        {Object.entries(result.summaries ?? {}).map(([key, value]) => <p key={key}>{key}: {value} (Canonical Units)</p>)}
      </details>
      <fieldset className="max-h-48 overflow-auto"><legend>Channels</legend>
        {Object.entries(result.channels).map(([key, channel], index) => <div key={key} className="flex gap-2">
          <label><input aria-label={`Select ${key}`} type="checkbox" checked={selected.includes(key)} onChange={(event) => setSelected(event.target.checked ? [...selected, key] : selected.filter((name) => name !== key))} /> {key} ({channel.unit})</label>
          <input aria-label={`Color for ${key}`} type="color" value={colors[key] ?? COLORS[index % COLORS.length]} onChange={(event) => setColors({ ...colors, [key]: event.target.value })} />
        </div>)}
      </fieldset>
      <div className="flex flex-wrap gap-3">
        <label>Y Minimum<input aria-label="Y Minimum" type="number" value={min} onChange={(event) => setMin(event.target.value)} placeholder="Auto" /></label>
        <label>Y Maximum<input aria-label="Y Maximum" type="number" value={max} onChange={(event) => setMax(event.target.value)} placeholder="Auto" /></label>
        <label><input type="checkbox" checked={showEvents} onChange={(event) => setShowEvents(event.target.checked)} /> Show Events</label>
      </div>
      {!validRange && <p role="alert">Y Minimum Must Be Less Than Y Maximum.</p>}
      <div ref={plots}>{units.map((axisUnit) => <div key={axisUnit} className="h-80" aria-label={`Biomechanics Plot (${axisUnit})`}>
        <ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{ bottom: 25, left: 20, right: 20 }}>
          <CartesianGrid stroke="#64748b" strokeDasharray="3 3" />
          <XAxis dataKey="time" type="number" domain={['dataMin', 'dataMax']} label={{ value: 'Time (s)', position: 'insideBottom', offset: -15 }} />
          <YAxis label={{ value: axisUnit, angle: -90, position: 'insideLeft' }} domain={validRange ? [min === '' ? 'auto' : Number(min), max === '' ? 'auto' : Number(max)] : ['auto', 'auto']} allowDataOverflow />
          <Tooltip /><Legend />
          {selected.filter((key) => result.channels[key].unit === axisUnit).map((key) => <Line key={key} dataKey={key} name={key} type="linear" connectNulls={false} dot={false} isAnimationActive={false} stroke={colors[key] ?? COLORS[Object.keys(result.channels).indexOf(key) % COLORS.length]} />)}
          {showEvents && Object.entries(result.events ?? {}).map(([name, time]) => <ReferenceLine key={name} x={time} label={name} strokeDasharray="4 4" />)}
        </LineChart></ResponsiveContainer>
      </div>)}</div>
      <button disabled={!selected.length} onClick={() => {
        const quote = (value: unknown) => `"${String(value ?? '').replace(/"/g, '""')}"`;
        const header = ['Time (s)', ...selected.map((key) => `${key} (${result.channels[key].unit})`)];
        const data = result.times.map((time, i) => [time, ...selected.map((key) => result.channels[key].values[i])]);
        download([header, ...data].map((row) => row.map(quote).join(',')).join('\r\n'), 'biomechanics.csv', 'text/csv');
      }}>Export CSV</button>
      <button disabled={!selected.length} onClick={() => {
        plots.current?.querySelectorAll('svg.recharts-surface').forEach((svg, index) => download(new XMLSerializer().serializeToString(svg), `biomechanics-${index + 1}.svg`, 'image/svg+xml'));
      }}>Export Plots (SVG)</button>
      <button onClick={() => download(JSON.stringify(raw, null, 2), 'biomechanics-result.json', 'application/json')}>Export Result JSON</button>
    </>}
  </section>;
}
