import { useEffect, useRef, useState } from 'react';
import type { SourceFitScope, SourceFitScopeBinding } from '@/api/necromatcher';
import { registerReviewedWindow } from '@/api/necromatcher';

export function SourceScopeSummary({scope, binding}: {scope?: SourceFitScope | null; binding?: SourceFitScopeBinding | null}) {
  if (!scope) return null;
  return <div className="text-xs"><p>Reviewed Original Frames: {scope.first_frame} to {scope.end_exclusive_frame} (Exclusive). {scope.review.reason}. {scope.review.uncertainty_policy}. Hand Contact and Physical Release Time Remain Unmeasured.</p>
    {binding && <p>Stored Fit Domain: Frames {binding.frame_indices[0]} to {binding.frame_indices[binding.frame_indices.length - 1]}; Source PTS {binding.first_pts.join('/')} to {binding.last_pts.join('/')}. Exact Source Clock; Physical Time Unqualified.</p>}
  </div>;
}

function reviewedWindow(value: unknown): value is SourceFitScope {
  if (!value || typeof value !== 'object') return false;
  const record = value as Partial<SourceFitScope>;
  return record.schema === 'necromatcher/source-fit-scope/1' &&
    typeof record.capture_id === 'string' && record.purpose === 'both_hands_on_club' &&
    Number.isInteger(record.first_frame) && Number.isInteger(record.end_exclusive_frame) &&
    Number(record.first_frame) >= 0 && Number(record.end_exclusive_frame) - Number(record.first_frame) >= 2 &&
    !!record.review && typeof record.review.reason === 'string' &&
    typeof record.review.uncertainty_policy === 'string' && record.review.contact_calibrated === false;
}

export function SourceScopeInput({fit, disabled, inherited, binding, onChange}: {
  fit?: string;
  disabled: boolean; inherited?: SourceFitScope | null; binding?: SourceFitScopeBinding | null;
  onChange: (record: SourceFitScope | null, blocked: boolean) => void;
}) {
  const [selected, setSelected] = useState<SourceFitScope | null>(null);
  const [error, setError] = useState('');
  const input = useRef<HTMLInputElement>(null);
  const generation = useRef(0);
  useEffect(() => () => {generation.current += 1;}, []);
  const remove = () => {
    generation.current += 1; setSelected(null); setError('');
    if (input.current) input.current.value = '';
    onChange(null, false);
  };
  const load = async (file: File) => {
    const current = ++generation.current;
    setSelected(null); setError(''); onChange(null, true);
    try {
      if (file.size > 1024 * 1024) throw new Error('Reviewed Fitting Window Must Be at Most 1 MiB.');
      let record: unknown = JSON.parse(await file.text());
      if (current !== generation.current) return;
      if (record && typeof record === 'object' && 'schema' in record && record.schema === 'necromatcher/source-fit-scope-review/1') {
        if (!fit) throw new Error('Select a Source Fit Before Importing a Raw Review.');
        record = await registerReviewedWindow(fit, file);
      }
      if (!reviewedWindow(record)) throw new Error('Choose a Reviewed Fitting Window Record.');
      if (current !== generation.current) return;
      if (inherited && (record.first_frame < inherited.first_frame || record.end_exclusive_frame > inherited.end_exclusive_frame)) throw new Error('Reviewed Fitting Window Cannot Widen the Inherited Window.');
      setSelected(record); onChange(record, false);
    } catch (reason) {
      if (current !== generation.current) return;
      setError(reason instanceof Error ? reason.message : 'Reviewed Fitting Window Could Not Be Read.');
      onChange(null, true);
    }
  };
  return <div className="space-y-2">
    <label className="block">Import Reviewed Window<input ref={input} type="file" accept=".json,application/json" disabled={disabled} onChange={(event) => {const file = event.target.files?.[0]; if (file) void load(file); else remove();}} /></label>
    <SourceScopeSummary scope={selected ?? inherited} binding={selected ? null : binding} />
    <p className="text-xs">Raw review receipts are registered as immutable library evidence. The server verifies original frames and the review receipt. Removing an imported window retains the inherited window; a descendant cannot widen it.</p>
    <button type="button" disabled={disabled} onClick={remove}>Remove Imported Window</button>
    {error && <p role="alert">{error}</p>}
  </div>;
}
