import { describe, expect, it, vi } from 'vitest';
import { render } from '@testing-library/react';
import { screen, fireEvent } from '@testing-library/dom';
import type { EngineComparisonViewModel } from './engineComparisonViewModel';
import { EngineComparisonPanel } from './EngineComparisonPanel';

const viewModel: EngineComparisonViewModel = {
  datasetLabel: 'Dataset: fixture',
  canCompare: true,
  selectedEngineNames: ['mujoco', 'drake'],
  options: [
    {
      name: 'mujoco',
      displayName: 'MuJoCo',
      support: 'ready',
      disabledReason: null,
      capabilities: ['rigid_body'],
    },
    {
      name: 'drake',
      displayName: 'Drake',
      support: 'ready',
      disabledReason: null,
      capabilities: ['rigid_body'],
    },
    {
      name: 'putting_green',
      displayName: 'Putting Green',
      support: 'blocked',
      disabledReason: 'No comparable rollout capability advertised',
      capabilities: ['terrain'],
    },
  ],
  columns: [
    {
      name: 'mujoco',
      displayName: 'MuJoCo',
      hasFrame: true,
      provenance: {
        engine: 'mujoco',
        version: '3.1.0',
        frame: 12,
        time: 0.24,
        capabilities: ['rigid_body'],
      },
      metrics: { 'state.qpos.rms': 1.2 },
    },
    {
      name: 'drake',
      displayName: 'Drake',
      hasFrame: false,
      provenance: {
        engine: 'drake',
        version: 'unknown',
        frame: null,
        time: null,
        capabilities: ['rigid_body'],
      },
      metrics: {},
    },
  ],
  annotations: [
    {
      metric: 'run data',
      baseline: 'mujoco',
      compared: 'drake',
      delta: null,
      severity: 'pending',
      label: 'Run each selected engine on this dataset to populate comparison data',
    },
  ],
  emptyMessage: 'Run each selected engine on this dataset to fill the comparison',
};

describe('EngineComparisonPanel', () => {
  it('renders comparison options, provenance, and divergence annotations', () => {
    render(<EngineComparisonPanel viewModel={viewModel} onToggleEngine={vi.fn()} />);

    expect(screen.getByRole('heading', { name: /engine comparison/i })).toBeInTheDocument();
    expect(screen.getByText('Dataset: fixture')).toBeInTheDocument();
    expect(screen.getByLabelText(/compare mujoco/i)).toBeChecked();
    expect(screen.getAllByText('Version')).toHaveLength(2);
    expect(screen.getByText('3.1.0')).toBeInTheDocument();
    expect(screen.getByText(/populate comparison data/i)).toBeInTheDocument();
  });

  it('toggles ready engines and disables unsupported options', () => {
    const onToggle = vi.fn();
    render(<EngineComparisonPanel viewModel={viewModel} onToggleEngine={onToggle} />);

    fireEvent.click(screen.getByLabelText(/compare mujoco/i));
    expect(onToggle).toHaveBeenCalledWith('mujoco');
    expect(screen.getByLabelText(/compare putting green/i)).toBeDisabled();
  });
});
