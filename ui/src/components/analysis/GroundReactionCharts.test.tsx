import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { GroundReactionCharts } from './GroundReactionCharts';
import * as api from '@/api/groundReaction';
import type { GroundReactionResponse } from '@/api/groundReaction';

vi.mock('@/api/fetch', () => ({ apiFetch: vi.fn() }));

const trace = (x: Array<number | null>) => ({
  x,
  y: x.map((v) => (v === null ? null : 0)),
  z: x.map((v) => (v === null ? null : 0)),
  magnitude: x.map((v) => (v === null ? null : Math.abs(v))),
});

const base: GroundReactionResponse = {
  run_id: 'r1',
  engine: 'mujoco',
  available: true,
  reason: null,
  time_s: [0, 0.01, 0.02],
  feet: ['left', 'right'],
  events: { impact: 0.02 },
  units: { left_force_n: 'N' },
  labels: {},
  load_share: {
    left: [0.5, 0.5, 0.5],
    right: [0.5, 0.5, 0.5],
  },
  traces: {
    left_force_n: trace([0, 10, 20]),
    right_force_n: trace([0, 10, 20]),
    net_force_n: trace([0, 20, 40]),
    left_cop_m: trace([0, 0.01, 0.02]),
    right_cop_m: trace([0, -0.01, -0.02]),
    net_cop_m: trace([0, 0, 0]),
    left_free_moment_nm: trace([0, 1, 2]),
    right_free_moment_nm: trace([0, -1, -2]),
    net_free_moment_nm: trace([0, 0, 0]),
    left_moment_com_nm: trace([0, 1, 2]),
    right_moment_com_nm: trace([0, -1, -2]),
    net_moment_com_nm: trace([0, 0, 0]),
  },
};

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('GroundReactionCharts', () => {
  it('renders the six ground-reaction panels with an impact marker', async () => {
    vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(base);
    render(<GroundReactionCharts runId="r1" />);
    expect(await screen.findByTestId('ground-reaction-chart-vertical_force')).toHaveTextContent(
      'Vertical Ground Reaction Force',
    );
    expect(screen.getByTestId('ground-reaction-chart-net_force')).toHaveTextContent(
      'Net Force Components',
    );
    expect(screen.getByTestId('ground-reaction-chart-load_share')).toHaveTextContent(
      'Vertical Load Share',
    );
    expect(screen.getByTestId('ground-reaction-chart-cop_path')).toHaveTextContent(
      'Centre of Pressure Path',
    );
    expect(screen.getByTestId('ground-reaction-chart-free_moment')).toHaveTextContent(
      'Free Moment About the Vertical',
    );
    expect(screen.getByTestId('ground-reaction-chart-net_moment')).toHaveTextContent(
      'Net Moment About CoM',
    );
    expect(screen.getByTestId('ground-reaction-event-vertical_force-impact')).toBeInTheDocument();
  });

  it('draws unavailable samples as gaps, never zero', async () => {
    const gappy: GroundReactionResponse = {
      ...base,
      traces: { ...base.traces, left_force_n: trace([10, null, 30]) },
    };
    vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(gappy);
    render(<GroundReactionCharts runId="r1" />);
    const left = await screen.findByTestId('ground-reaction-line-vertical_force-Left');
    // two separate runs: the unavailable middle sample breaks the line
    expect((left.getAttribute('d') ?? '').match(/M/g)).toHaveLength(2);
  });

  it('shows "Unavailable" and the reason when the run has no ground contact', async () => {
    vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue({
      ...base,
      available: false,
      reason: 'Simscape until GCV-3',
      time_s: [],
      feet: [],
      traces: {},
      load_share: {},
    });
    render(<GroundReactionCharts runId="r1" />);
    const unavailable = await screen.findByTestId('ground-reaction-unavailable');
    expect(unavailable).toHaveTextContent('Unavailable');
    expect(unavailable).toHaveTextContent('Simscape until GCV-3');
    expect(screen.queryByTestId('ground-reaction-chart-vertical_force')).toBeNull();
  });

  it('plots vertical force in N when no body weight is known', async () => {
    vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(base);
    render(<GroundReactionCharts runId="r1" />);
    expect(await screen.findByTestId('ground-reaction-chart-vertical_force')).toHaveTextContent(
      '(N)',
    );
  });

  it('plots vertical force in BW once net_force_bw is present', async () => {
    const bw: GroundReactionResponse = {
      ...base,
      traces: {
        ...base.traces,
        net_force_bw: trace([0, 1, 2]),
        left_force_bw: trace([0, 0.5, 1]),
        right_force_bw: trace([0, 0.5, 1]),
      },
    };
    vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(bw);
    render(<GroundReactionCharts runId="r1" />);
    expect(await screen.findByTestId('ground-reaction-chart-vertical_force')).toHaveTextContent(
      '(BW)',
    );
  });

  it('builds the request path', () => {
    expect(api.buildGroundReactionPath({ runId: 'r1', impactTimeS: 0.9 })).toBe(
      '/api/analysis/ground-reaction?run_id=r1&impact_time_s=0.9',
    );
    expect(api.buildGroundReactionPath({})).toBe('/api/analysis/ground-reaction');
  });
});
