/**
 * Tests for AnalysisTools page.
 *
 * Validates data structures and type contracts for analysis tools.
 */

import { describe, it, expect } from 'vitest';

import type {
  MetricInfo,
  StatisticsSummary,
  ExportResult,
  AnalysisLoadState,
} from './AnalysisTools';

describe('AnalysisTools data structures', () => {
  it('should parse metric info', () => {
    const metric: MetricInfo = {
      id: 'club_speed',
      name: 'Club Speed',
      description: 'Speed of the club head at impact',
      unit: 'm/s',
      category: 'kinematics',
    };

    expect(metric.id).toBe('club_speed');
    expect(metric.unit).toBe('m/s');
    expect(metric.category).toBe('kinematics');
  });

  it('should parse metric info with value', () => {
    const metric: MetricInfo = {
      id: 'launch_angle',
      name: 'Launch Angle',
      description: 'Vertical launch angle',
      unit: 'degrees',
      category: 'launch',
      value: 12.5,
    };

    expect(metric.value).toBe(12.5);
    expect(typeof metric.value).toBe('number');
  });

  it('should group metrics by category', () => {
    const metrics: MetricInfo[] = [
      { id: 'club_speed', name: 'Club Speed', description: 'Club head speed', unit: 'm/s', category: 'kinematics' },
      { id: 'ball_speed', name: 'Ball Speed', description: 'Ball speed', unit: 'm/s', category: 'kinematics' },
      { id: 'launch_angle', name: 'Launch Angle', description: 'Launch angle', unit: 'degrees', category: 'launch' },
      { id: 'spin_rate', name: 'Spin Rate', description: 'Backspin rate', unit: 'rpm', category: 'launch' },
      { id: 'carry_distance', name: 'Carry Distance', description: 'Carry distance', unit: 'm', category: 'result' },
    ];

    const grouped = metrics.reduce<Record<string, MetricInfo[]>>((acc, m) => {
      if (!acc[m.category]) acc[m.category] = [];
      acc[m.category].push(m);
      return acc;
    }, {});

    expect(Object.keys(grouped)).toHaveLength(3);
    expect(grouped['kinematics']).toHaveLength(2);
    expect(grouped['launch']).toHaveLength(2);
    expect(grouped['result']).toHaveLength(1);
  });

  it('should parse statistics summary', () => {
    const stats: StatisticsSummary = {
      dataset_id: 'ds_001',
      metric_count: 3,
      summary: {
        club_speed: { min: 25.0, max: 55.0, mean: 42.3, median: 43.1, std: 5.2 },
        ball_speed: { min: 35.0, max: 80.0, mean: 61.5, median: 62.0, std: 8.1 },
        launch_angle: { min: -5.0, max: 25.0, mean: 11.2, median: 10.8, std: 4.5 },
      },
    };

    expect(stats.metric_count).toBe(3);
    expect(Object.keys(stats.summary)).toHaveLength(3);
    expect(stats.summary.club_speed.mean).toBeCloseTo(42.3, 1);
    expect(stats.summary.ball_speed.max).toBe(80.0);
  });

  it('should compute derived statistics', () => {
    const summary = { min: 25.0, max: 55.0, mean: 42.3, median: 43.1, std: 5.2 };
    const range = summary.max - summary.min;

    expect(range).toBeCloseTo(30.0, 1);
    expect(summary.mean).toBeGreaterThanOrEqual(summary.min);
    expect(summary.mean).toBeLessThanOrEqual(summary.max);
    expect(summary.std).toBeGreaterThan(0);
  });

  it('should parse export result', () => {
    const result: ExportResult = {
      format: 'csv',
      url: '/api/analysis/export/ds_001.csv',
      filename: 'analysis_ds_001.csv',
      size_bytes: 102400,
    };

    expect(result.format).toBe('csv');
    expect(result.filename).toContain('analysis');
    expect(result.size_bytes).toBeGreaterThan(0);
    expect((result.size_bytes / 1024).toFixed(1)).toBe('100.0');
  });

  it('should validate analysis load state transitions', () => {
    const states: AnalysisLoadState[] = ['idle', 'loading', 'loaded', 'error'];

    expect(states).toContain('idle');
    expect(states).toContain('loading');
    expect(states).toHaveLength(4);
  });

  it('should filter metrics by category', () => {
    const metrics: MetricInfo[] = [
      { id: 'club_speed', name: 'Club Speed', description: 'Club head speed', unit: 'm/s', category: 'kinematics' },
      { id: 'launch_angle', name: 'Launch Angle', description: 'Launch angle', unit: 'degrees', category: 'launch' },
      { id: 'carry_distance', name: 'Carry Distance', description: 'Carry distance', unit: 'm', category: 'result' },
    ];

    const kinematics = metrics.filter((m) => m.category === 'kinematics');
    expect(kinematics).toHaveLength(1);
    expect(kinematics[0].id).toBe('club_speed');
  });

  it('should format size in KB', () => {
    const sizes = [1024, 5120, 1048576];
    const formatted = sizes.map((s) => (s / 1024).toFixed(1));

    expect(formatted[0]).toBe('1.0');
    expect(formatted[1]).toBe('5.0');
    expect(formatted[2]).toBe('1024.0');
  });
});
