/**
 * LoopNullSpacePanel tests (NSX-44, issue #12576).
 *
 * Uses the shared cross-UI golden plot-model fixture (`golden_plot_model_v1.json`)
 * and verifies preset selection, partial-invariants polling, and DELETE job cancellation.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
  LoopNullSpacePanel,
  NSX_HELP_GUIDE_PATH,
  type NSXPlotModel,
} from '../LoopNullSpacePanel';
import goldenPlotModelJson from '../../../../../src/tools/grip_wrench_plots/golden_plot_model_v1.json';

const GOLDEN_MODEL = goldenPlotModelJson as unknown as NSXPlotModel;

describe('LoopNullSpacePanel', () => {
  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it('renders all 4 panels, evidence badges, segments, and help link from the shared golden fixture', () => {
    render(<LoopNullSpacePanel initialPlotModel={GOLDEN_MODEL} />);

    expect(screen.getByTestId('nsx-panel-invariants')).toBeTruthy();
    expect(screen.getByTestId('nsx-panel-envelopes')).toBeTruthy();
    expect(screen.getByTestId('nsx-panel-hand_split')).toBeTruthy();
    expect(screen.getByTestId('nsx-panel-diagnostics')).toBeTruthy();

    // Invariants & provenance badge
    expect(screen.getByTestId('nsx-invariant-bare_club_net_wrench')).toBeTruthy();
    expect(screen.getByTestId('nsx-badge-bare_club_net_wrench').textContent).toContain(
      'MODEL_CONDITIONED_INVARIANT',
    );

    // Envelopes with INDETERMINATE and DETERMINED_SIGN segments, binding hypotheses, and overlaid representative
    expect(screen.getByTestId('nsx-band-lead_shoulder_torque')).toBeTruthy();
    expect(screen.getByTestId('nsx-rep-lead_shoulder_torque')).toBeTruthy();
    expect(screen.getByTestId('nsx-segment-lead_shoulder_torque-0').textContent).toContain(
      'INDETERMINATE',
    );
    expect(screen.getByTestId('nsx-segment-lead_shoulder_torque-1').textContent).toContain(
      'H-GRIP-FRICTION-CONE',
    );

    // Hand split s1..s6
    expect(screen.getByTestId('nsx-band-s1_axial_internal_force_n')).toBeTruthy();
    expect(screen.getByTestId('nsx-band-s6_bending_w_diff_moment_nm')).toBeTruthy();

    // Diagnostics
    expect(screen.getByTestId('nsx-diag-rank_jc').textContent).toContain('6, 6, 6');

    // Help link
    const helpLink = screen.getByTestId('nsx-help-link') as HTMLAnchorElement;
    expect(helpLink.getAttribute('href')).toBe(NSX_HELP_GUIDE_PATH);
  });

  it('switches presets via the registry-backed selector', () => {
    render(<LoopNullSpacePanel initialPlotModel={GOLDEN_MODEL} />);
    const select = screen.getByTestId('nsx-preset-select') as HTMLSelectElement;
    expect(select.value).toBe('capacity_min_effort');

    fireEvent.change(select, { target: { value: 'envelope_only' } });
    expect(select.value).toBe('envelope_only');

    fireEvent.change(select, { target: { value: 'smooth_capacity' } });
    expect(select.value).toBe('smooth_capacity');
  });

  it('creates an API job, renders partial invariants first while running, and settles on completed plot model', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        status: 202,
        json: async () => ({
          job_id: 'job-nsx-1',
          status: 'running',
          progress_percent: 35,
          partial_invariants: {
            bare_club_net_wrench: {
              label: 'Bare-Club Net Hand Wrench Magnitude',
              units: 'N',
              values: [149.666295, 152.166295, 154.666295],
              allocation_dependence: 'INVARIANT',
            },
          },
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job-nsx-1',
          status: 'completed',
          progress_percent: 100,
          plot_model: GOLDEN_MODEL,
        }),
      });

    render(<LoopNullSpacePanel pollIntervalMs={10} />);
    fireEvent.click(screen.getByTestId('nsx-run-button'));

    await waitFor(() => {
      expect(screen.getByTestId('nsx-job-status').textContent).toContain('completed');
    });

    expect(mockFetch).toHaveBeenCalledWith(
      '/analysis/loop-nullspace/jobs',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(mockFetch).toHaveBeenCalledWith(
      '/analysis/loop-nullspace/jobs/job-nsx-1',
      expect.objectContaining({ method: 'GET' }),
    );
    expect(screen.getByTestId('nsx-panel-envelopes')).toBeTruthy();
  });

  it('shows partial invariants while running and cancels the active job via DELETE', async () => {
    let resolvePoll: ((val: unknown) => void) | null = null;
    mockFetch.mockImplementation((url: string, init?: RequestInit) => {
      if (init?.method === 'POST') {
        return Promise.resolve({
          ok: true,
          status: 202,
          json: async () => ({
            job_id: 'job-cancel-99',
            status: 'running',
            partial_invariants: {
              bare_club_net_wrench: {
                label: 'Bare-Club Net Hand Wrench Magnitude',
                units: 'N',
                values: [149.666295],
              },
            },
          }),
        });
      }
      if (init?.method === 'DELETE' && String(url).endsWith('/job-cancel-99')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            job_id: 'job-cancel-99',
            status: 'cancelled',
          }),
        });
      }
      return new Promise((resolve) => {
        resolvePoll = resolve;
      });
    });

    render(<LoopNullSpacePanel pollIntervalMs={50} />);
    fireEvent.click(screen.getByTestId('nsx-run-button'));

    await waitFor(() => {
      expect(screen.getByTestId('nsx-partial-invariants')).toBeTruthy();
    });

    const cancelBtn = screen.getByTestId('nsx-cancel-button');
    fireEvent.click(cancelBtn);

    await waitFor(() => {
      expect(screen.getByTestId('nsx-job-status').textContent).toContain('cancelled');
    });
    expect(mockFetch).toHaveBeenCalledWith(
      '/analysis/loop-nullspace/jobs/job-cancel-99',
      expect.objectContaining({ method: 'DELETE' }),
    );
    if (resolvePoll) {
      resolvePoll({ ok: true, status: 200, json: async () => ({ status: 'cancelled' }) });
    }
  });
});
