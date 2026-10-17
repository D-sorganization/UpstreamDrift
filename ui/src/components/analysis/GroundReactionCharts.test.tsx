import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { GroundReactionCharts } from './GroundReactionCharts';
import * as api from '@/api/groundReaction';
import type { GroundReactionResponse } from '@/api/groundReaction';
import { fillPlateForm, submitPlateForm } from './plateImportForm.fixture';

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

  it('keeps the force namespace on force event labels', async () => {
    vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue({
      ...base,
      events: { impact: 0.02, 'force.max_trail': 0.01 },
    });
    render(<GroundReactionCharts runId="r1" />);
    const marker = await screen.findByTestId(
      'ground-reaction-event-vertical_force-force.max_trail',
    );
    expect(marker).toHaveTextContent('force.max_trail');
    expect(marker).not.toHaveTextContent(/^max_trail$/);
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

  describe('measured plate overlay', () => {
    const unobserved = [null, null, null];
    const measured = (z: Array<number | null>) => ({
      x: unobserved,
      y: unobserved,
      z,
      magnitude: unobserved,
    });
    const withOverlay = (qualified: boolean | null): GroundReactionResponse => ({
      ...base,
      measured_qualified: qualified,
      observability: { fx: false, fy: false, fz: true, cop: true, free_moment: false },
      traces: {
        ...base.traces,
        measured_left_force_n: measured([5, 15, 25]),
        measured_left_cop_m: {
          x: [0, 0.01, null],
          y: [0, 0.02, null],
          z: unobserved,
          magnitude: unobserved,
        },
      },
    });

    it('draws measured series dashed beside the model with the unqualified banner', async () => {
      vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(withOverlay(false));
      render(<GroundReactionCharts runId="r1" />);
      expect(await screen.findByTestId('ground-reaction-unqualified-banner')).toHaveTextContent(
        'Unqualified measurement',
      );
      const line = screen.getByTestId('ground-reaction-line-vertical_force-Measured Left');
      expect(line.getAttribute('stroke-dasharray')).toBe('4 2');
      const model = screen.getByTestId('ground-reaction-line-vertical_force-Left');
      expect(model.getAttribute('stroke-dasharray')).toBeNull();
      const cop = screen.getByTestId('ground-reaction-line-cop_path-Measured Left');
      expect(cop.getAttribute('stroke-dasharray')).toBe('4 2');
    });

    const withBand = (): GroundReactionResponse => ({
      ...withOverlay(false),
      measured_uncertainty_source: 'datasheet rev 3',
      traces: {
        ...withOverlay(false).traces,
        measured_left_force_sigma_n: measured([1, 2, null]),
      },
    });

    it('shades the measured load by its 1-sigma uncertainty and names the source', async () => {
      vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(withBand());
      render(<GroundReactionCharts runId="r1" />);
      const band = await screen.findByTestId('ground-reaction-band-vertical_force-Measured Left');
      expect(band.getAttribute('d')).toMatch(/Z$/);
      expect(band.getAttribute('fill-opacity')).toBe('0.2');
      expect(screen.getByTestId('ground-reaction-uncertainty-caption')).toHaveTextContent(
        'Shaded: measured load \u00b11\u03c3 (declared uncertainty: datasheet rev 3)',
      );
      expect(screen.getByText('Measured Left \u00b11\u03c3')).toBeInTheDocument();
    });

    it('draws no band or caption when the overlay carries no sigma trace', async () => {
      vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(withOverlay(false));
      render(<GroundReactionCharts runId="r1" />);
      await screen.findByTestId('ground-reaction-line-vertical_force-Measured Left');
      expect(screen.queryByTestId('ground-reaction-band-vertical_force-Measured Left')).toBeNull();
      expect(screen.queryByTestId('ground-reaction-uncertainty-caption')).toBeNull();
    });

    it('hides the banner for a qualified measurement and for plain runs', async () => {
      vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(withOverlay(true));
      const { unmount } = render(<GroundReactionCharts runId="r1" />);
      await screen.findByTestId('ground-reaction-line-vertical_force-Measured Left');
      expect(screen.queryByTestId('ground-reaction-unqualified-banner')).toBeNull();
      unmount();
      vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(base);
      render(<GroundReactionCharts runId="r1" />);
      await screen.findByTestId('ground-reaction-chart-vertical_force');
      expect(screen.queryByTestId('ground-reaction-unqualified-banner')).toBeNull();
      expect(screen.queryByTestId('ground-reaction-line-vertical_force-Measured Left')).toBeNull();
    });

    it('posts the overlay request when an overlay is given', async () => {
      const post = vi
        .spyOn(api, 'fetchGroundReactionOverlay')
        .mockResolvedValue(withOverlay(false));
      const get = vi.spyOn(api, 'fetchGroundReaction');
      const request: api.PlateOverlayRequest = {
        trial: '20260101-120000-123.json',
        handedness: 'RH',
        origin_m: [0, 0, 0],
        declared: {
          force_floor_n: 2,
          force_fraction: 0.01,
          cop_m: 0.01,
          sensing_depth_sigma_m: 0.002,
          uncertainty_source: 'declared',
          sensing_depth_m: 0.015,
          capacity_n: 3923,
        },
        alignment: { offset_s: 0, sigma_offset_s: 0.001, kind: 'manual', note: 'clock' },
      };
      render(<GroundReactionCharts runId="r1" overlay={request} />);
      await screen.findByTestId('ground-reaction-unqualified-banner');
      expect(post).toHaveBeenCalledWith({ ...request, run_id: 'r1' }, expect.any(AbortSignal));
      expect(get).not.toHaveBeenCalled();
    });

    it('overlays the trial declared in the form', async () => {
      vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(base);
      const post = vi
        .spyOn(api, 'fetchGroundReactionOverlay')
        .mockResolvedValue(withOverlay(false));
      render(<GroundReactionCharts runId="r1" />);
      await screen.findByTestId('ground-reaction-chart-vertical_force');
      fillPlateForm();
      submitPlateForm();
      await screen.findByTestId('ground-reaction-unqualified-banner');
      const [request] = post.mock.calls[0];
      expect(request.run_id).toBe('r1');
      expect(request.trial).toBe('session-1/swing-0001.json');
      expect(request.declared.capacity_n).toBe(3923);
    });

    it('shows a server refusal next to the form and returns to the model', async () => {
      const get = vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(base);
      vi.spyOn(api, 'fetchGroundReactionOverlay').mockRejectedValue(
        new Error('capacity_n must be positive'),
      );
      render(<GroundReactionCharts runId="r1" />);
      await screen.findByTestId('ground-reaction-chart-vertical_force');
      fillPlateForm();
      submitPlateForm();
      expect(await screen.findByTestId('plate-import-error')).toHaveTextContent(
        'Plate overlay refused: capacity_n must be positive',
      );
      await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
      expect(screen.queryByRole('alert', { name: /Ground reaction unavailable/ })).toBeNull();
      expect(await screen.findByTestId('ground-reaction-chart-vertical_force')).toBeTruthy();
    });

    it('does not draw measured load in BW mode (the plate has no model body weight)', async () => {
      const bw = withOverlay(false);
      bw.traces = {
        ...bw.traces,
        net_force_bw: trace([0, 1, 2]),
        left_force_bw: trace([0, 1, 2]),
      };
      vi.spyOn(api, 'fetchGroundReaction').mockResolvedValue(bw);
      render(<GroundReactionCharts runId="r1" />);
      await screen.findByTestId('ground-reaction-unqualified-banner');
      expect(screen.queryByTestId('ground-reaction-line-vertical_force-Measured Left')).toBeNull();
    });
  });

  it('builds the request path', () => {
    expect(api.buildGroundReactionPath({ runId: 'r1', impactTimeS: 0.9 })).toBe(
      '/api/analysis/ground-reaction?run_id=r1&impact_time_s=0.9',
    );
    expect(api.buildGroundReactionPath({})).toBe('/api/analysis/ground-reaction');
  });
});
