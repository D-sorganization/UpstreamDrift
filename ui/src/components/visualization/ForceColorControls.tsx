import { useId, useState } from 'react';
import { validateForceColorScale } from './forceColors';
import type { ForceColorScale } from './forceColors';

interface Props {
  scale: ForceColorScale;
  onChange: (scale: ForceColorScale) => void;
}

/** Reusable display settings; hosts persist validated values and own load data. */
export function ForceColorControls({ scale, onChange }: Props) {
  const id = useId();
  const [edits, setEdits] = useState<Partial<ForceColorScale>>({});
  const [error, setError] = useState('');
  const draft = { ...scale, ...edits };
  const apply = () => {
    try {
      onChange(validateForceColorScale({ ...draft, enabled: scale.enabled }));
      setEdits({});
      setError('');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Invalid settings');
    }
  };
  return <fieldset className="space-y-2 p-3">
    <legend>Segment Force Colors</legend>
    <label><input type="checkbox" checked={scale.enabled}
      onChange={event => onChange({ ...scale, enabled: event.target.checked })} />
      Color Segments by Axial Force</label>
    {([
      ['tension_limit_n', 'Tension Saturation (N)'],
      ['compression_limit_n', 'Compression Saturation (N)'],
      ['deadband_n', 'Neutral Band ± (N)'],
    ] as const).map(([key, label]) => <div key={key}>
      <label htmlFor={`${id}-${key}`}>{label}</label>
      <input id={`${id}-${key}`} type="number" min="0" step="any" value={draft[key]}
        onChange={event => setEdits({ ...edits, [key]: event.target.value === '' ? NaN : Number(event.target.value) })} />
    </div>)}
    {([
      ['tension_color', 'Tension Color'], ['compression_color', 'Compression Color'],
      ['neutral_color', 'Neutral Color'],
    ] as const).map(([key, label]) => <div key={key}>
      <label htmlFor={`${id}-${key}`}>{label}</label>
      <input id={`${id}-${key}`} type="color" value={draft[key]}
        onChange={event => setEdits({ ...edits, [key]: event.target.value })} />
    </div>)}
    <button type="button" onClick={apply}>Apply Colors and Ranges</button>
    {error && <p role="alert">{error}</p>}
    <p>
      <span style={{ color: scale.compression_color }} aria-hidden>■</span> Compression ≤ −{scale.compression_limit_n} N ·{' '}
      <span style={{ color: scale.neutral_color }} aria-hidden>■</span> Neutral ±{scale.deadband_n} N ·{' '}
      <span style={{ color: scale.tension_color }} aria-hidden>■</span> Tension ≥ +{scale.tension_limit_n} N
    </p>
    <p>Positive = tension; negative = compression. Values clip at the limits.
      Unavailable samples keep their original color. Axial force is not stress.</p>
  </fieldset>;
}
