import { describe, expect, it } from 'vitest';
import type { SimulationFrame } from '@/api/client';
import type { ManagedEngine } from '@/stores/useEngineStore';
import {
  buildEngineComparisonOptions,
  buildEngineComparisonViewModel,
  coerceComparisonSelection,
  engineSupportsComparison,
  toggleComparisonEngine,
} from './engineComparisonViewModel';

const engine = (overrides: Partial<ManagedEngine>): ManagedEngine => ({
  name: 'mujoco',
  displayName: 'MuJoCo',
  description: 'Physics engine',
  loadState: 'loaded',
  available: true,
  capabilities: ['rigid_body'],
  ...overrides,
});

const frame = (
  overrides: Partial<SimulationFrame> = {},
): SimulationFrame => ({
  frame: 10,
  time: 0.2,
  state: { qpos: [1, 2], energy: [4] },
  analysis: { joint_angles: [0.1, 0.2] },
  ...overrides,
});

describe('engineComparisonViewModel', () => {
  it('treats advertised rollout-style capabilities as comparable', () => {
    expect(engineSupportsComparison(engine({ capabilities: ['forward_simulation'] }))).toBe(
      true,
    );
    expect(engineSupportsComparison(engine({ capabilities: ['terrain'] }))).toBe(false);
  });

  it('builds capability-aware options that block unloaded and unsupported engines', () => {
    const options = buildEngineComparisonOptions([
      engine({ name: 'mujoco', displayName: 'MuJoCo' }),
      engine({
        name: 'drake',
        displayName: 'Drake',
        loadState: 'idle',
        capabilities: ['rigid_body'],
      }),
      engine({
        name: 'putting_green',
        displayName: 'Putting Green',
        capabilities: ['terrain'],
      }),
    ]);

    expect(options.find((option) => option.name === 'mujoco')?.support).toBe('ready');
    expect(options.find((option) => option.name === 'drake')?.support).toBe('pending');
    expect(options.find((option) => option.name === 'putting_green')?.support).toBe(
      'blocked',
    );
  });

  it('coerces selection to currently ready engines', () => {
    const options = buildEngineComparisonOptions([
      engine({ name: 'mujoco' }),
      engine({ name: 'drake', loadState: 'idle' }),
    ]);

    expect(coerceComparisonSelection(['mujoco', 'drake'], options)).toEqual(['mujoco']);
  });

  it('does not toggle unsupported engines into the comparison', () => {
    const options = buildEngineComparisonOptions([
      engine({ name: 'mujoco' }),
      engine({ name: 'drake', loadState: 'idle' }),
    ]);

    expect(toggleComparisonEngine(['mujoco'], 'drake', options)).toEqual(['mujoco']);
    expect(toggleComparisonEngine(['mujoco'], 'mujoco', options)).toEqual([]);
  });

  it('builds side-by-side columns with provenance and divergence annotations', () => {
    const viewModel = buildEngineComparisonViewModel({
      datasetLabel: 'Dataset: fixture',
      engines: [
        engine({ name: 'mujoco', displayName: 'MuJoCo', version: '3.1.0' }),
        engine({ name: 'drake', displayName: 'Drake', capabilities: ['rigid_body'] }),
      ],
      selectedEngineNames: ['mujoco', 'drake'],
      framesByEngine: {
        mujoco: frame(),
        drake: frame({
          frame: 11,
          time: 0.22,
          state: { qpos: [1.2, 2.1], energy: [4.4] },
          analysis: { joint_angles: [0.1, 0.3] },
        }),
      },
    });

    expect(viewModel.canCompare).toBe(true);
    expect(viewModel.columns).toHaveLength(2);
    expect(viewModel.columns[0].provenance.version).toBe('3.1.0');
    expect(viewModel.annotations.length).toBeGreaterThan(0);
    expect(viewModel.annotations.some((annotation) => annotation.severity === 'critical')).toBe(
      true,
    );
  });

  it('surfaces pending annotations when selected engines have no captured frame', () => {
    const viewModel = buildEngineComparisonViewModel({
      datasetLabel: 'Dataset: fixture',
      engines: [engine({ name: 'mujoco' }), engine({ name: 'drake' })],
      selectedEngineNames: ['mujoco', 'drake'],
      framesByEngine: {
        mujoco: frame(),
      },
    });

    expect(viewModel.emptyMessage).toMatch(/run each selected engine/i);
    expect(viewModel.annotations[0].severity).toBe('pending');
  });
});
