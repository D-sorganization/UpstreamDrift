import { describe, expect, it } from 'vitest';
import {
  ALIGNMENT_KINDS,
  PLATE_FORM_FIELDS,
  initialPlateFormTexts,
  requestFromPlateForm,
} from './plateImportForm';

const complete = (changes: Record<string, string> = {}): Record<string, string> => ({
  ...initialPlateFormTexts(),
  trial: 'session-1/swing-0001.json',
  device: 'Lab pair A',
  uncertainty_source: 'datasheet rev 3',
  force_floor_n: '2.0',
  force_fraction: '0.01',
  cop_m: '0.010',
  sensing_depth_m: '0.015',
  sensing_depth_sigma_m: '0.002',
  capacity_n: '3923',
  origin_m: '1.0, 2.0, 0.0',
  offset_s: '0.25',
  sigma_offset_s: '0.001',
  alignment_note: 'clap on both clocks',
  ...changes,
});

describe('plate import form', () => {
  it('starts with every physical fact empty and only conventions prefilled', () => {
    const texts = initialPlateFormTexts();
    for (const field of PLATE_FORM_FIELDS) {
      if (['handedness', 'target_dir', 'alignment_kind'].includes(field.key)) continue;
      expect(texts[field.key], field.key).toBe('');
    }
    expect(texts.handedness).toBe('RH');
    expect(texts.target_dir).toBe('0, -1, 0');
    expect(texts.alignment_kind).toBe('manual');
  });

  it('offers cross-correlation alongside the declared alignments', () => {
    expect(ALIGNMENT_KINDS).toEqual(['manual', 'shot_impact', 'absolute_utc', 'cross_correlation']);
  });

  it('sends the model a cross-correlated offset was fitted against', () => {
    const result = requestFromPlateForm(
      complete({ alignment_kind: 'cross_correlation', derived_from_model: 'model-a' }),
    );
    expect(result.error).toBeNull();
    expect(result.request?.alignment.kind).toBe('cross_correlation');
    expect(result.request?.alignment.derived_from_model).toBe('model-a');
  });

  it('refuses a cross-correlation without its model, and a model without one', () => {
    for (const changes of [
      { alignment_kind: 'cross_correlation' },
      { derived_from_model: 'model-a' },
    ]) {
      expect(requestFromPlateForm(complete(changes)).error?.key).toBe('derived_from_model');
    }
  });

  it('builds the overlay request from a complete form', () => {
    const result = requestFromPlateForm(complete());
    expect(result.error).toBeNull();
    expect(result.request).toEqual({
      trial: 'session-1/swing-0001.json',
      handedness: 'RH',
      origin_m: [1, 2, 0],
      target_dir: [0, -1, 0],
      declared: {
        force_floor_n: 2,
        force_fraction: 0.01,
        cop_m: 0.01,
        sensing_depth_sigma_m: 0.002,
        uncertainty_source: 'datasheet rev 3',
        sensing_depth_m: 0.015,
        capacity_n: 3923,
        device: 'Lab pair A',
      },
      alignment: {
        offset_s: 0.25,
        sigma_offset_s: 0.001,
        kind: 'manual',
        note: 'clap on both clocks',
        derived_from_model: null,
      },
    });
  });

  it('leaves a blank device label to the server default', () => {
    const result = requestFromPlateForm(complete({ device: '  ' }));
    expect(result.request?.declared.device).toBeNull();
  });

  it('names the topmost missing field and builds no request', () => {
    const result = requestFromPlateForm(initialPlateFormTexts());
    expect(result.request).toBeNull();
    expect(result.error).toEqual({ key: 'trial', message: 'required' });
    const later = requestFromPlateForm(complete({ capacity_n: '' }));
    expect(later.error?.key).toBe('capacity_n');
  });

  it.each([
    ['capacity_n', 'lots'],
    ['force_floor_n', 'NaN'],
    ['offset_s', 'Infinity'],
    ['origin_m', '1, 2'],
    ['target_dir', '0, -1, x'],
    ['handedness', 'ambi'],
    ['alignment_kind', 'fitted'],
  ])('refuses %s = %s with that field named', (key, text) => {
    const result = requestFromPlateForm(complete({ [key]: text }));
    expect(result.request).toBeNull();
    expect(result.error?.key).toBe(key);
  });

  it('refuses a trial path that is absolute or climbs out of the root', () => {
    for (const trial of ['/etc/passwd', '../outside.json', 'a\\b.json']) {
      expect(requestFromPlateForm(complete({ trial })).error?.key, trial).toBe('trial');
    }
  });
});
