/**
 * Py<->TS metadata round-trip (epic #5968, DRY guarantee).
 *
 * The generated `fieldMetadata.ts` must exactly reflect the YAML at
 * `configs/ux/field_metadata.yaml`, which is the single source of
 * truth.  This test re-parses the YAML independently and asserts the
 * generated TS registry matches, so editing the YAML without
 * regenerating fails CI.
 */

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { parse } from 'yaml';
import { describe, it, expect } from 'vitest';
import { FIELD_METADATA, getFieldMetadata } from './fieldMetadata';

const YAML_PATH = resolve(__dirname, '../../../configs/ux/field_metadata.yaml');

interface RawField {
  id: string;
  label: string;
  short_help: string;
  long_help: string;
  units: string | null;
  valid_range: unknown;
  default: unknown;
  default_source: string;
  consumers?: string[];
  producers?: string[];
  example?: string;
}

function loadYaml(): RawField[] {
  const text = readFileSync(YAML_PATH, 'utf-8');
  const doc = parse(text) as { fields: RawField[] };
  return doc.fields;
}

describe('fieldMetadata round-trip', () => {
  it('covers every YAML field with matching ids', () => {
    const yamlIds = loadYaml()
      .map((f) => f.id)
      .sort();
    const tsIds = FIELD_METADATA.map((f) => f.id).sort();
    expect(tsIds).toEqual(yamlIds);
  });

  it('preserves help copy, range and provenance per field', () => {
    for (const raw of loadYaml()) {
      const ts = getFieldMetadata(raw.id);
      expect(ts.label).toBe(raw.label);
      expect(ts.shortHelp).toBe(raw.short_help);
      expect(ts.longHelp).toBe(raw.long_help);
      expect(ts.units).toBe(raw.units ?? null);
      expect(ts.defaultSource).toBe(raw.default_source);
      expect(ts.consumers).toEqual(raw.consumers ?? []);
      expect(ts.producers).toEqual(raw.producers ?? []);
      expect(ts.validRange).toEqual(raw.valid_range ?? null);
    }
  });

  it('covers the ParameterPanel and ActuatorPanel fields', () => {
    const ids = FIELD_METADATA.map((f) => f.id);
    expect(ids).toContain('simulation.duration');
    expect(ids).toContain('simulation.timestep');
    expect(ids).toContain('actuator.control_type');
    expect(ids).toContain('actuator.value');
    expect(ids).toContain('actuator.min_value');
    expect(ids).toContain('actuator.max_value');
  });

  it('throws for an unknown field id', () => {
    expect(() => getFieldMetadata('does.not.exist')).toThrow(/unknown field id/);
  });
});
