import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { GripWrenchCharts } from './GripWrenchCharts';
import * as api from '@/api/gripWrench';
import type { GripWrenchResponse } from '@/api/gripWrench';

vi.mock('@/api/fetch', () => ({ apiFetch: vi.fn() }));

const trace = (x: Array<number | null>) => ({
  x,
  y: x.map((v) => (v === null ? null : 0)),
  z: x.map((v) => (v === null ? null : 0)),
  magnitude: x.map((v) => (v === null ? null : Math.abs(v))),
});

const base: GripWrenchResponse = {
  run_id: 'r1',
  engine: 'mujoco',
  available: true,
  reason: null,
  time_s: [0, 0.01, 0.02],
  split_method: 'efc_force',
  split_method_by_sample: ['efc_force', 'efc_force', 'efc_force'],
  unavailable_reasons: ['', '', ''],
  events: { impact: 0.02 },
  units: { left_force_n: 'N', couple_nm: 'N*m' },
  labels: {
    left_force_n: 'Left Hand Force',
    right_force_n: 'Right Hand Force',
    net_force_n: 'Net Force at Midpoint',
    couple_nm: 'Couple at Midpoint (World)',
    couple_local_nm: 'Couple at Midpoint (Club)',
    contact_force_moment_nm: 'Contact Force Moment',
    applied_free_torque_nm: 'Applied Free Torque',
  },
  traces: {
    left_force_n: trace([0, 10, 20]),
    right_force_n: trace([0, -10, -20]),
    net_force_n: trace([0, 0, 0]),
    couple_nm: trace([0, 2, 4]),
    couple_local_nm: trace([0, 2, 4]),
    contact_force_moment_nm: trace([0, 2, 4]),
    applied_free_torque_nm: trace([0, 0, 0]),
  },
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('GripWrenchCharts', () => {
  it('shows the split method and one chart per quantity', async () => {
    vi.spyOn(api, 'fetchGripWrench').mockResolvedValue(base);
    render(<GripWrenchCharts runId="r1" />);
    expect(await screen.findByTestId('grip-split-method')).toHaveTextContent('efc_force');
    expect(screen.getByTestId('grip-chart-hand_forces')).toBeInTheDocument();
    expect(screen.getByTestId('grip-chart-net_force')).toBeInTheDocument();
    expect(screen.getByTestId('grip-chart-couple')).toBeInTheDocument();
    expect(screen.getByTestId('grip-chart-couple_split')).toBeInTheDocument();
    expect(screen.getByTestId('grip-impact-marker-hand_forces')).toBeInTheDocument();
  });

  it('draws unavailable samples as gaps, never zero', async () => {
    const gappy: GripWrenchResponse = {
      ...base,
      split_method: 'mixed',
      split_method_by_sample: ['efc_force', 'allocation', 'efc_force'],
      traces: { ...base.traces, left_force_n: trace([10, null, 30]) },
    };
    vi.spyOn(api, 'fetchGripWrench').mockResolvedValue(gappy);
    render(<GripWrenchCharts runId="r1" />);
    const left = await screen.findByTestId('grip-line-hand_forces-left_force_n');
    // two separate runs: the unavailable middle sample breaks the line
    expect((left.getAttribute('d') ?? '').match(/M/g)).toHaveLength(2);
    expect(screen.getByTestId('grip-split-method')).toHaveTextContent('mixed');
    expect(screen.getByTestId('grip-unavailable-note')).toHaveTextContent('unavailable');
  });

  it('shows the reason when the run has no grip data', async () => {
    vi.spyOn(api, 'fetchGripWrench').mockResolvedValue({
      ...base,
      available: false,
      reason: 'run carries no grip wrench',
      time_s: [],
      traces: {},
      split_method: 'unavailable',
    });
    render(<GripWrenchCharts runId="r1" />);
    expect(await screen.findByTestId('grip-unavailable')).toHaveTextContent(
      'run carries no grip wrench',
    );
    expect(screen.queryByTestId('grip-chart-couple')).toBeNull();
  });

  it('switches the couple frame between world and club', async () => {
    const spy = vi.spyOn(api, 'fetchGripWrench').mockResolvedValue({
      ...base,
      traces: { ...base.traces, couple_local_nm: trace([0, 5, 9]) },
    });
    render(<GripWrenchCharts runId="r1" />);
    await waitFor(() => expect(spy).toHaveBeenCalled());
    await screen.findByTestId('grip-chart-couple');
    expect(screen.getByTestId('grip-line-couple-couple_nm')).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Couple frame'), { target: { value: 'club' } });
    const club = await screen.findByTestId('grip-line-couple-couple_local_nm');
    expect(club).toBeInTheDocument();
    expect(screen.queryByTestId('grip-line-couple-couple_nm')).toBeNull();
    expect(screen.getByTestId('grip-chart-couple')).toHaveTextContent('Club');
  });

  it('builds the request path', () => {
    expect(api.buildGripWrenchPath({ runId: 'r1', impactTimeS: 0.9 })).toBe(
      '/api/analysis/grip-wrench?run_id=r1&impact_time_s=0.9',
    );
    expect(api.buildGripWrenchPath({})).toBe('/api/analysis/grip-wrench');
  });
});
