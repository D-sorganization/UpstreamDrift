/**
 * Web plate declarations form for the ground reaction overlay (OF-07b, #13122).
 *
 * Presentation over `plateImportForm`: one row per field, grouped by section,
 * like the PyQt6 Plate Trial Declarations dialog. A field the form refuses is
 * named under it; `serverError` shows the plate-overlay route's refusal.
 */
import { useState, type FormEvent } from 'react';
import type { PlateOverlayRequest } from '@/api/groundReaction';
import {
  PLATE_FORM_FIELDS,
  initialPlateFormTexts,
  requestFromPlateForm,
  type PlateFormField,
  type PlateFormSection,
} from './plateImportForm';

const SECTION_TITLES: Record<PlateFormSection, string> = {
  Trial: 'Plate Trial',
  Plate: 'Plate Declarations',
  Stance: 'Stance Placement',
  Clock: 'Clock Alignment',
};

const INPUT_CLASS =
  'w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-xs text-gray-200';

function FieldInput({
  field,
  value,
  onChange,
}: {
  field: PlateFormField;
  value: string;
  onChange: (value: string) => void;
}) {
  const id = `plate-field-${field.key}`;
  return (
    <label className="block text-xs text-gray-400" htmlFor={id}>
      {field.label}
      {field.choices ? (
        <select
          id={id}
          data-testid={id}
          className={INPUT_CLASS}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        >
          {field.choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          data-testid={id}
          className={INPUT_CLASS}
          value={value}
          placeholder={field.required ? 'required' : undefined}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </label>
  );
}

export function PlateImportForm({
  onSubmit,
  onClear,
  serverError = null,
  active = false,
}: {
  /** Receives the request once the form declares every required fact. */
  onSubmit: (request: PlateOverlayRequest) => void;
  /** Removes the current overlay; shown only while `active`. */
  onClear?: () => void;
  /** The plate-overlay route's refusal of the last request, if any. */
  serverError?: string | null;
  active?: boolean;
}) {
  const [texts, setTexts] = useState<Record<string, string>>(initialPlateFormTexts);
  const [formError, setFormError] = useState<string | null>(null);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    const result = requestFromPlateForm(texts);
    if (result.error) {
      setFormError(`${result.error.key}: ${result.error.message}`);
      return;
    }
    setFormError(null);
    onSubmit(result.request);
  };

  const error = formError ?? (serverError ? `Plate overlay refused: ${serverError}` : null);
  const sections = Object.keys(SECTION_TITLES) as PlateFormSection[];
  return (
    <form className="mt-3 space-y-2" data-testid="plate-import-form" onSubmit={submit} noValidate>
      {sections.map((section) => (
        <fieldset key={section} className="border border-gray-700 rounded p-2">
          <legend className="text-xs font-semibold text-gray-300 px-1">
            {SECTION_TITLES[section]}
          </legend>
          <div className="grid grid-cols-2 gap-2">
            {PLATE_FORM_FIELDS.filter((f) => f.section === section).map((f) => (
              <FieldInput
                key={f.key}
                field={f}
                value={texts[f.key] ?? ''}
                onChange={(value) => setTexts((prev) => ({ ...prev, [f.key]: value }))}
              />
            ))}
          </div>
        </fieldset>
      ))}
      {error && (
        <div role="alert" className="text-xs text-red-400" data-testid="plate-import-error">
          {error}
        </div>
      )}
      <div className="flex gap-2">
        <button type="submit" className="px-3 py-1 text-xs rounded bg-blue-600 text-white">
          Overlay Plate Trial
        </button>
        {active && onClear && (
          <button
            type="button"
            className="px-3 py-1 text-xs rounded bg-gray-700 text-gray-200"
            onClick={onClear}
          >
            Clear Overlay
          </button>
        )}
      </div>
    </form>
  );
}
