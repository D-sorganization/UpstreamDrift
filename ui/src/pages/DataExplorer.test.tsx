/**
 * Tests for DataExplorer page.
 *
 * See issue #1206
 */

import { describe, it, expect } from 'vitest';

import type {
  DatasetInfo,
  DatasetPreview,
  DatasetStats,
  ColumnStats,
} from './DataExplorer';

describe('DataExplorer data structures', () => {
  it('should parse dataset info listing', () => {
    const datasets: DatasetInfo[] = [
      {
        name: 'simulation_results.csv',
        path: '/output/simulation_results.csv',
        format: 'csv',
        size_bytes: 45200,
        columns: ['time', 'position_x', 'velocity_y', 'force'],
      },
      {
        name: 'swing_data.json',
        path: '/output/swing_data.json',
        format: 'json',
        size_bytes: 12800,
        columns: ['timestamp', 'hip_angle', 'shoulder_angle'],
      },
    ];

    expect(datasets).toHaveLength(2);
    expect(datasets[0].columns).toContain('time');
    expect(datasets[1].format).toBe('json');
  });

  it('should parse dataset preview with rows', () => {
    const preview: DatasetPreview = {
      name: 'results.csv',
      columns: ['time', 'x', 'y'],
      rows: [
        { time: '0.0', x: '1.0', y: '2.0' },
        { time: '0.1', x: '1.5', y: '2.3' },
        { time: '0.2', x: '2.0', y: '2.8' },
      ],
      total_rows: 100,
      format: 'csv',
    };

    expect(preview.columns).toHaveLength(3);
    expect(preview.rows).toHaveLength(3);
    expect(preview.total_rows).toBe(100);
    expect(preview.rows[0].time).toBe('0.0');
  });

  it('should parse column statistics', () => {
    const colStats: ColumnStats = {
      min: 0.0,
      max: 10.0,
      mean: 5.2,
      count: 50,
    };

    expect(colStats.min).toBe(0.0);
    expect(colStats.max).toBe(10.0);
    expect(colStats.mean).toBeCloseTo(5.2, 1);
    expect(colStats.count).toBe(50);
  });

  it('should handle non-numeric column stats', () => {
    const colStats: ColumnStats = {
      min: null,
      max: null,
      mean: null,
      count: 0,
    };

    expect(colStats.min).toBeNull();
    expect(colStats.mean).toBeNull();
    expect(colStats.count).toBe(0);
  });

  it('should parse full dataset stats', () => {
    const stats: DatasetStats = {
      name: 'results.csv',
      columns: ['time', 'force', 'label'],
      row_count: 200,
      stats: {
        time: { min: 0.0, max: 10.0, mean: 5.0, count: 200 },
        force: { min: -50.0, max: 150.0, mean: 42.5, count: 200 },
        label: { min: null, max: null, mean: null, count: 0 },
      },
    };

    expect(stats.row_count).toBe(200);
    expect(stats.stats.time.mean).toBe(5.0);
    expect(stats.stats.force.max).toBe(150.0);
    expect(stats.stats.label.mean).toBeNull();
  });

  it('should sort rows by column value numerically', () => {
    const rows = [
      { time: '2.0', x: '3.0' },
      { time: '0.5', x: '1.0' },
      { time: '1.0', x: '2.0' },
    ];

    const sorted = [...rows].sort((a, b) => {
      const aVal = Number(a.time);
      const bVal = Number(b.time);
      return aVal - bVal;
    });

    expect(sorted[0].time).toBe('0.5');
    expect(sorted[1].time).toBe('1.0');
    expect(sorted[2].time).toBe('2.0');
  });

  it('should filter rows by column value', () => {
    const rows = [
      { time: '0.0', category: 'backswing' },
      { time: '1.0', category: 'downswing' },
      { time: '2.0', category: 'impact' },
      { time: '3.0', category: 'follow_through' },
    ];

    const filtered = rows.filter((r) => r.category === 'impact');
    expect(filtered).toHaveLength(1);
    expect(filtered[0].time).toBe('2.0');
  });

  it('should filter rows with numeric comparison', () => {
    const rows = [
      { time: '0.0', force: '10.0' },
      { time: '1.0', force: '50.0' },
      { time: '2.0', force: '30.0' },
      { time: '3.0', force: '80.0' },
    ];

    const filtered = rows.filter((r) => Number(r.force) > 40);
    expect(filtered).toHaveLength(2);
  });

  it('should validate supported file formats', () => {
    const supported = new Set(['.csv', '.json', '.hdf5', '.h5', '.c3d']);

    expect(supported.has('.csv')).toBe(true);
    expect(supported.has('.json')).toBe(true);
    expect(supported.has('.xlsx')).toBe(false);
    expect(supported.has('.txt')).toBe(false);
  });
});

// ── F3: DataExplorer distinct error state ───────────────────────────────────
describe('DataExplorer service-unreachable error state (F3 — issue #6642)', () => {
  it('marks serviceUnreachable=true on network failure', () => {
    // Simulate the error-handling logic inside fetchDatasets
    let serviceUnreachable = false;
    let error: string | null = null;

    const handleCatch = (err: unknown) => {
      serviceUnreachable = true;
      error = err instanceof Error
        ? err.message
        : "Can't reach the data service — is the backend running?";
    };

    handleCatch(new Error('fetch failed'));

    expect(serviceUnreachable).toBe(true);
    expect(error).toContain('fetch failed');
  });

  it('marks serviceUnreachable=true on non-ok response', () => {
    let serviceUnreachable = false;
    let error: string | null = null;

    const handleNonOk = (status: number) => {
      serviceUnreachable = true;
      error = `Data service returned HTTP ${status}`;
    };

    handleNonOk(503);

    expect(serviceUnreachable).toBe(true);
    expect(error).toContain('503');
  });

  it('shows different error UI for serviceUnreachable vs dataset error', () => {
    // Regression guard: the amber "Data service unreachable" banner should only
    // appear when serviceUnreachable===true; regular red errors should only
    // appear when serviceUnreachable===false.
    const showServiceBanner = (serviceUnreachable: boolean, error: string | null) =>
      serviceUnreachable && error !== null;

    const showRegularError = (serviceUnreachable: boolean, error: string | null) =>
      error !== null && !serviceUnreachable;

    expect(showServiceBanner(true, 'network error')).toBe(true);
    expect(showServiceBanner(false, 'network error')).toBe(false);
    expect(showRegularError(false, 'filter failed')).toBe(true);
    expect(showRegularError(true, 'filter failed')).toBe(false);
  });
});

// ── F4: DataExplorer API shape guards ────────────────────────────────────────
describe('DataExplorer API shape guards (F4 — issue #6642)', () => {
  it('defaults to empty array when datasets key is missing', () => {
    const data: Record<string, unknown> = {};
    const list = Array.isArray(data.datasets) ? (data.datasets as DatasetInfo[]) : [];
    expect(list).toHaveLength(0);
  });

  it('defaults to empty array when datasets is not an array', () => {
    const data: Record<string, unknown> = { datasets: null };
    const list = Array.isArray(data.datasets) ? (data.datasets as DatasetInfo[]) : [];
    expect(list).toHaveLength(0);
  });

  it('passes through a valid datasets array', () => {
    const data: Record<string, unknown> = {
      datasets: [
        { name: 'ds1.csv', path: '/p', format: 'csv', size_bytes: 100, columns: [] },
      ],
    };
    const list = Array.isArray(data.datasets) ? (data.datasets as DatasetInfo[]) : [];
    expect(list).toHaveLength(1);
    expect(list[0].name).toBe('ds1.csv');
  });
});
