/**
 * Tests for MatchedSwings Results page (MS-85, #10358).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { MatchedSwingsPage } from './MatchedSwings';
import type { MatchedSwingLedgerResponse } from '@/api/matchedSwings';

vi.mock('@/components/visualization/MocapSkeleton3D', () => ({
  default: () => <div data-testid="mocap-3d">3D preview</div>,
}));

const SAMPLE_LEDGER: MatchedSwingLedgerResponse = {
  schema_version: 'matched-swing-api/1',
  total: 2,
  runs: [
    {
      id: 'aaa111',
      engine: 'drake',
      lane: 'matched',
      capture: 'driver',
      candidate_sha256: 'cand1',
      receipt_sha256: 'aaa111',
      horizon_s: 0.85,
      verdict: 'PASSED',
      metrics: { whole_marker_rmse_m: 0.022 },
      capabilities: {
        has_candidate_npz: true,
        has_animation_gif: true,
        has_parity_report: true,
        candidate_profile: 'kinematic',
        horizon_s: 0.85,
      },
    },
    {
      id: 'bbb222',
      engine: 'opensim',
      lane: 'matched',
      capture: 'driver',
      candidate_sha256: 'cand2',
      receipt_sha256: 'bbb222',
      horizon_s: 0.85,
      verdict: 'REJECTED',
      metrics: { whole_marker_rmse_m: 0.245 },
      capabilities: {
        has_candidate_npz: false,
        has_animation_gif: false,
        has_parity_report: false,
        candidate_profile: null,
        horizon_s: 0.85,
      },
    },
  ],
};

describe('MatchedSwingsPage', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo) => {
        const url = String(input);
        if (url.includes('/matched-swings/aaa111/candidate')) {
          return new Response(
            JSON.stringify({
              frame_index: 0,
              frame_count: 1,
              joints: [{ name: 'pelvis', position: [0, 0, 1], confidence: 1, parent: null }],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          );
        }
        if (url.includes('/matched-swings')) {
          return new Response(JSON.stringify(SAMPLE_LEDGER), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response('not found', { status: 404 });
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('renders run list with verdict badges', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText('Matched Swing Results')).toBeInTheDocument();
    expect(await screen.findByText('PASSED')).toBeInTheDocument();
    expect(screen.getByText('REJECTED')).toBeInTheDocument();
    expect(screen.getByText(/drake/i)).toBeInTheDocument();
  });

  it('filters runs by engine', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByText('opensim');
    const engineSelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(engineSelect, 'drake');
    await waitFor(() => {
      expect(screen.queryByText('opensim')).not.toBeInTheDocument();
    });
  });

  it('loads 3D preview for selected run with candidate npz', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    expect(await screen.findByTestId('mocap-3d')).toBeInTheDocument();
  });

  it('links to cross-engine dashboard route', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    const link = await screen.findByRole('link', { name: /cross-engine dashboard/i });
    expect(link).toHaveAttribute('href', '/tools/cross-engine');
  });
});
