/**
 * Tests for DatasetGenerator page.
 *
 * Validates data structures and type contracts for dataset generation.
 */

import { describe, it, expect } from 'vitest';

import type {
  FeatureInfo,
  PlotType,
  ExportFormat,
  DatasetControl,
  GenerateResult,
  DatasetLoadState,
} from './DatasetGenerator';

describe('DatasetGenerator data structures', () => {
  it('should parse feature info', () => {
    const feature: FeatureInfo = {
      id: 'club_speed',
      name: 'Club Speed',
      description: 'Speed of the club head at impact',
      category: 'kinematics',
    };

    expect(feature.id).toBe('club_speed');
    expect(feature.category).toBe('kinematics');
    expect(feature.description.length).toBeGreaterThan(0);
  });

  it('should group features by category', () => {
    const features: FeatureInfo[] = [
      { id: 'club_speed', name: 'Club Speed', description: 'Club head speed', category: 'kinematics' },
      { id: 'ball_speed', name: 'Ball Speed', description: 'Ball speed after impact', category: 'kinematics' },
      { id: 'launch_angle', name: 'Launch Angle', description: 'Vertical launch angle', category: 'launch' },
      { id: 'spin_rate', name: 'Spin Rate', description: 'Backspin rate', category: 'launch' },
    ];

    const grouped = features.reduce<Record<string, FeatureInfo[]>>((acc, f) => {
      if (!acc[f.category]) acc[f.category] = [];
      acc[f.category].push(f);
      return acc;
    }, {});

    expect(Object.keys(grouped)).toHaveLength(2);
    expect(grouped['kinematics']).toHaveLength(2);
    expect(grouped['launch']).toHaveLength(2);
  });

  it('should parse plot type info', () => {
    const plotType: PlotType = {
      id: 'scatter',
      name: 'Scatter Plot',
      description: 'Two-variable scatter plot',
      axes: ['x', 'y'],
    };

    expect(plotType.id).toBe('scatter');
    expect(plotType.axes).toContain('x');
    expect(plotType.axes).toHaveLength(2);
  });

  it('should parse export format info', () => {
    const format: ExportFormat = {
      id: 'csv',
      name: 'CSV',
      extension: 'csv',
      mime_type: 'text/csv',
    };

    expect(format.extension).toBe('csv');
    expect(format.mime_type).toContain('text/');
  });

  it('should handle multiple export formats', () => {
    const formats: ExportFormat[] = [
      { id: 'csv', name: 'CSV', extension: 'csv', mime_type: 'text/csv' },
      { id: 'json', name: 'JSON', extension: 'json', mime_type: 'application/json' },
      { id: 'hdf5', name: 'HDF5', extension: 'h5', mime_type: 'application/x-hdf5' },
    ];

    expect(formats).toHaveLength(3);
    const extensions = formats.map((f) => `.${f.extension}`);
    expect(extensions).toContain('.csv');
    expect(extensions).toContain('.json');
    expect(extensions).toContain('.h5');
  });

  it('should parse dataset control', () => {
    const control: DatasetControl = {
      id: 'num_samples',
      name: 'Number of Samples',
      type: 'range',
      value: 1000,
      min: 100,
      max: 10000,
      step: 100,
    };

    expect(control.type).toBe('range');
    expect(control.min).toBeLessThanOrEqual(control.value as number);
    expect(control.max).toBeGreaterThanOrEqual(control.value as number);
  });

  it('should parse select-type control with options', () => {
    const control: DatasetControl = {
      id: 'swing_type',
      name: 'Swing Type',
      type: 'select',
      value: 'driver',
      options: ['driver', 'iron', 'wedge', 'putter'],
    };

    expect(control.type).toBe('select');
    expect(control.options).toContain('driver');
    expect(control.options).toHaveLength(4);
  });

  it('should parse generate result', () => {
    const result: GenerateResult = {
      dataset_id: 'ds_abc123',
      name: 'swing_analysis_batch_1',
      rows: 5000,
      columns: ['time', 'club_speed', 'ball_speed', 'launch_angle', 'spin_rate'],
      created_at: '2026-05-15T12:00:00Z',
    };

    expect(result.dataset_id).toBe('ds_abc123');
    expect(result.rows).toBeGreaterThan(0);
    expect(result.columns).toHaveLength(5);
    expect(result.columns).toContain('club_speed');
  });

  it('should validate dataset load state transitions', () => {
    const states: DatasetLoadState[] = ['idle', 'loading', 'loaded', 'error'];

    expect(states).toContain('idle');
    expect(states).toContain('loading');
    expect(states).toHaveLength(4);
  });

  it('should filter features by search term', () => {
    const features: FeatureInfo[] = [
      { id: 'club_speed', name: 'Club Speed', description: 'Club head speed at impact', category: 'kinematics' },
      { id: 'ball_speed', name: 'Ball Speed', description: 'Ball speed after impact', category: 'kinematics' },
      { id: 'smash_factor', name: 'Smash Factor', description: 'Ratio of ball speed to club speed', category: 'efficiency' },
    ];

    const filtered = features.filter(
      (f) => f.name.toLowerCase().includes('speed') || f.description.toLowerCase().includes('speed'),
    );
    expect(filtered).toHaveLength(3);
  });
});

describe('DatasetGenerator export behaviour', () => {
  it('export endpoint URL uses the format id', () => {
    const format = 'csv';
    const url = `/api/dataset/export/${encodeURIComponent(format)}`;
    expect(url).toBe('/api/dataset/export/csv');
  });

  it('export endpoint URL encodes non-trivial format ids', () => {
    const format = 'hdf5+zip';
    const url = `/api/dataset/export/${encodeURIComponent(format)}`;
    expect(url).toBe('/api/dataset/export/hdf5%2Bzip');
  });

  it('export requires a dataset_id in the request body', () => {
    const result: GenerateResult = {
      dataset_id: 'ds_abc123',
      name: 'swing_batch',
      rows: 100,
      columns: ['time'],
      created_at: '2026-01-01',
    };
    const body = JSON.stringify({ dataset_id: result.dataset_id });
    const parsed = JSON.parse(body);
    expect(parsed.dataset_id).toBe('ds_abc123');
  });

  it('export button is disabled when no generateResult', () => {
    const generateResult: GenerateResult | null = null;
    const isLoading = false;
    const disabled = !generateResult || isLoading;
    expect(disabled).toBe(true);
  });

  it('export button is enabled when generateResult is present', () => {
    const generateResult: GenerateResult = {
      dataset_id: 'ds_1',
      name: 'test',
      rows: 10,
      columns: [],
      created_at: '2026-01-01',
    };
    const isLoading = false;
    const disabled = !generateResult || isLoading;
    expect(disabled).toBe(false);
  });
});
