import { useEffect, useRef, useState } from 'react';

type Props = {onChange: (record: Record<string, unknown> | null, blocked: boolean) => void; disabled?: boolean};
const maximumBytes = 2 * 1024 * 1024;

/** Preview only; the server rebinds every reviewed frame to canonical source bytes. */
export function ShaftEvidenceInput({onChange, disabled}: Props) {
  const generation = useRef(0);
  const input = useRef<HTMLInputElement>(null);
  const [summary, setSummary] = useState('');
  const [error, setError] = useState('');
  useEffect(() => () => {generation.current += 1;}, []);

  const remove = () => {
    generation.current += 1;
    if (input.current) input.current.value = '';
    setSummary(''); setError(''); onChange(null, false);
  };
  const load = async (file: File) => {
    const selected = ++generation.current;
    setSummary('Reading Evidence…'); setError(''); onChange(null, true);
    try {
      if (file.size > maximumBytes) throw new Error('Evidence File Exceeds 2 MiB.');
      const record: unknown = JSON.parse(await file.text());
      if (!record || typeof record !== 'object' || Array.isArray(record) ||
          !('schema' in record) || record.schema !== 'necromatcher/shaft-axis-evidence/1' ||
          !('capture_id' in record) || typeof record.capture_id !== 'string' ||
          !('frames' in record) || !Array.isArray(record.frames) || !record.frames.length) {
        throw new Error('Choose a Reviewed Shaft Evidence Record.');
      }
      if (generation.current !== selected) return;
      setSummary(`${record.frames.length} Reviewed Frame${record.frames.length === 1 ? '' : 's'} · Capture: ${record.capture_id}`);
      onChange(record as Record<string, unknown>, false);
    } catch (reason) {
      if (generation.current !== selected) return;
      setSummary(''); setError(reason instanceof Error ? reason.message : 'Evidence Could Not Be Read.');
      onChange(null, true);
    }
  };

  return <div className="space-y-2">
    <label className="block">Reviewed Shaft Evidence<input ref={input} type="file" accept=".json,application/json" disabled={disabled} onChange={(event) => {const selected = event.target.files?.[0]; if (selected) void load(selected); else remove();}} /></label>
    <p className="text-xs">Optional visible shaft fragments in original image pixels. Weights remain authored and uncalibrated. The server checks the selected fit, source frames and timing before starting.</p>
    {summary && <p role="status" className="text-xs">{summary}</p>}
    <button type="button" disabled={disabled} className="rounded border p-2" onClick={remove}>Remove Shaft Evidence</button>
    {error && <p role="alert">{error}</p>}
  </div>;
}
