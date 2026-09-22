/**
 * Tests for MatchedSwings Results page (MS-85, #10358).
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
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

const { fetchCandidatePreviewFrameMock } = vi.hoisted(() => ({
  fetchCandidatePreviewFrameMock: vi.fn(async () => ({
    id: 'aaa111',
    frame_index: 0,
    frame_count: 1,
    joints: [{ name: 'pelvis', position: [0, 0, 1], confidence: 1, parent: null }],
  })),
}));

vi.mock('@/api/matchedSwings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/matchedSwings')>();
  return {
    ...actual,
    fetchMatchedSwingLedger: vi.fn(async () => SAMPLE_LEDGER),
    fetchCandidatePreviewFrame: fetchCandidatePreviewFrameMock,
    matchedSwingAnimationUrl: (id: string) => `/api/v1/matched-swings/${id}/animation.gif`,
  };
});

describe('MatchedSwingsPage', () => {
  beforeEach(() => {
    fetchCandidatePreviewFrameMock.mockClear();
  });

  it('renders run list with verdict badges', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText('Matched Swing Results')).toBeInTheDocument();
    expect((await screen.findAllByText('PASSED')).length).toBeGreaterThan(0);
    expect((await screen.findAllByText('REJECTED')).length).toBeGreaterThan(0);
    expect(screen.getAllByRole('button', { name: /drake/i }).length).toBeGreaterThan(0);
  });

  it('filters runs by engine', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /opensim/i });
    const engineSelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(engineSelect, 'drake');
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /opensim/i })).not.toBeInTheDocument();
    });
  });

  it('loads 3D preview for selected run with candidate npz', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(fetchCandidatePreviewFrameMock).toHaveBeenCalledWith('aaa111', 0);
    });
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
