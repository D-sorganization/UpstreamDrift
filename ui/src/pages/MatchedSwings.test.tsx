/**
 * Tests for MatchedSwings Results page (MS-85, #10358).
 */

import { vi } from 'vitest';

declare const describe: (name: string, fn: () => void) => void;
declare const it: {
  (name: string, fn: () => void | Promise<void>): void;
  skip: (name: string, fn: () => void | Promise<void>) => void;
  only: (name: string, fn: () => void | Promise<void>) => void;
};
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const expect: any;
declare const beforeEach: (fn: () => void | Promise<void>) => void;
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router';
import { MatchedSwingsPage } from './MatchedSwings';
import { fetchMatchedSwingLedger, type MatchedSwingLedgerResponse } from '@/api/matchedSwings';

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
        has_video_export: true,
        has_animation_gif: true,
        has_parity_report: true,
        candidate_profile: 'kinematic',
        horizon_s: 0.85,
      },
      gates: [
        { name: 'g1', status: 'PASS', measured: 0.01, threshold: 0.02, unit: 'm' },
      ],
    },
    {
      id: 'bbb222',
      engine: 'opensim',
      lane: 'tour_matching',
      capture: 'iron',
      candidate_sha256: 'cand2',
      receipt_sha256: 'bbb222',
      horizon_s: 0.85,
      verdict: 'REJECTED',
      metrics: { whole_marker_rmse_m: 0.245 },
      capabilities: {
        has_candidate_npz: false,
        has_video_export: false,
        has_animation_gif: false,
        has_parity_report: false,
        candidate_profile: null,
        horizon_s: 0.85,
      },
      reason: 'unique_rejection_marker',
      qualification_note: 'preferred qualification note',
      gates: [],
    },
  ],
};

const {
  fetchCandidatePreviewFrameMock,
  fetchMatchedSwingReceiptMock,
  fetchParityReportMock,
  fetchMatchedSwingAnimationInfoMock,
  launchMatchedSwingNativeViewerMock,
} = vi.hoisted(() => ({
  fetchCandidatePreviewFrameMock: vi.fn(async () => ({
    id: 'aaa111',
    frame_index: 0,
    frame_count: 1,
    joints: [{ name: 'pelvis', position: [0, 0, 1], confidence: 1, parent: null }],
  })),
  fetchMatchedSwingReceiptMock: vi.fn(async () => ({
    id: 'aaa111',
    receipt: { engine: 'drake' },
  })),
  fetchParityReportMock: vi.fn(async () => ({
    schema_version: 'matched-swing-parity-report-v1',
  })),
  fetchMatchedSwingAnimationInfoMock: vi.fn(async () => ({
    schema_version: 'matched-swing-animation/1',
    frame_count: 1,
    durations_ms: [100],
    width: 1,
    height: 1,
  })),
  launchMatchedSwingNativeViewerMock: vi.fn(async (runId: string) => ({
    schema_version: 'matched-swing-native-viewer/1',
    run_id: runId,
    success: true,
    backend: 'meshcat',
    url: 'http://localhost:7000',
    message: 'Native viewer meshcat launched successfully',
  })),
}));

vi.mock('@/api/matchedSwings', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/matchedSwings')>();
  return {
    ...actual,
    fetchMatchedSwingLedger: vi.fn(async () => SAMPLE_LEDGER),
    fetchCandidatePreviewFrame: fetchCandidatePreviewFrameMock,
    fetchMatchedSwingReceipt: fetchMatchedSwingReceiptMock,
    fetchParityReport: fetchParityReportMock,
    launchMatchedSwingNativeViewer: launchMatchedSwingNativeViewerMock,
    matchedSwingAnimationUrl: (id: string) => `/api/v1/matched-swings/${id}/animation.gif`,
    fetchMatchedSwingAnimationInfo: fetchMatchedSwingAnimationInfoMock,
    matchedSwingAnimationFrameUrl: (id: string, index: number) =>
      `/api/v1/matched-swings/${id}/animation/frames/${index}`,
  };
});

describe('MatchedSwingsPage', () => {
  beforeEach(() => {
    fetchCandidatePreviewFrameMock.mockClear();
    fetchMatchedSwingReceiptMock.mockClear();
    fetchParityReportMock.mockClear();
    launchMatchedSwingNativeViewerMock.mockClear();
    vi.mocked(fetchMatchedSwingLedger).mockClear();
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

  it('filters runs by capture', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /opensim/i });
    const captureSelect = screen.getAllByRole('combobox')[1];
    await user.selectOptions(captureSelect, 'iron');
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /drake/i })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /opensim/i })).toBeInTheDocument();
    });
  });

  it('filters runs by lane', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /opensim/i });
    const laneSelect = screen.getAllByRole('combobox')[2];
    await user.selectOptions(laneSelect, 'tour_matching');
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /drake/i })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /opensim/i })).toBeInTheDocument();
    });
  });

  it('finds runs by reason text in search', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    const searchInput = screen.getByPlaceholderText(/engine, sha, id, reason/i);
    await user.type(searchInput, 'unique_rejection_marker');
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /drake/i })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: /opensim/i })).toBeInTheDocument();
    });
  });

  it('resets all filters and search', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /opensim/i });
    const engineSelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(engineSelect, 'drake');
    const searchInput = screen.getByPlaceholderText(/engine, sha, id, reason/i);
    await user.type(searchInput, 'cand1');
    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /opensim/i })).not.toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /^reset$/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /drake/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /opensim/i })).toBeInTheDocument();
    });
    expect(searchInput).toHaveValue('');
  });

  it('renders physical gates for the selected run and the empty-state placeholder', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    expect(await screen.findByText(/g1/)).toBeInTheDocument();

    await user.click(await screen.findByRole('button', { name: /opensim/i }));
    await waitFor(() => {
      expect(
        screen.getByText('No physical gates evaluated for this run.'),
      ).toBeInTheDocument();
    });
  });

  it('fetches and displays receipt JSON', async () => {
    const user = userEvent.setup();
    const { container } = render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    await user.click(screen.getByRole('button', { name: /view receipt json/i }));

    await waitFor(() => {
      expect(fetchMatchedSwingReceiptMock).toHaveBeenCalledWith('aaa111');
    });
    await waitFor(() => {
      expect(container.textContent).toContain('"engine": "drake"');
    });
  });

  it('shows output-policy badges beside the receipt without withheld numbers (#10382)', async () => {
    fetchMatchedSwingReceiptMock.mockResolvedValueOnce({
      id: 'aaa111',
      receipt: { outputs: { face_angle: { value: 7.77, uncertainty: 0.1 } } },
      output_policy: {
        schema_version: 'output-policy/1',
        outputs: [
          { name: 'face_angle', status: 'withheld', label: 'NOT VALIDATED', reason: 'not validated' },
        ],
        face_observability: null,
      },
    } as never);
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );
    await screen.findByRole('button', { name: /drake/i });
    await user.click(screen.getByRole('button', { name: /view receipt json/i }));
    expect(await screen.findByTestId('output-status-face_angle')).toHaveTextContent('NOT VALIDATED');
    expect(screen.queryByTestId('output-value-face_angle')).toBeNull();
  });

  it('links the export report action to the selected run report URL', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    const reportLink = screen.getByRole('link', { name: /^export report$/i });
    expect(reportLink).toHaveAttribute(
      'href',
      '/api/v1/matched-swings/aaa111/report',
    );
  });

  it('links the PDF report action to the format=pdf report URL', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    const pdfLink = screen.getByRole('link', { name: /export report \(pdf\)/i });
    expect(pdfLink).toHaveAttribute(
      'href',
      '/api/v1/matched-swings/aaa111/report?format=pdf',
    );
  });

  it('enables Export Video links for a run with a candidate NPZ', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    const gifLink = screen.getByRole('link', { name: /export video \(gif\)/i });
    const mp4Link = screen.getByRole('link', { name: /export video \(mp4\)/i });
    expect(gifLink).toHaveAttribute(
      'href',
      '/api/v1/matched-swings/aaa111/video?format=gif',
    );
    expect(mp4Link).toHaveAttribute(
      'href',
      '/api/v1/matched-swings/aaa111/video?format=mp4',
    );
  });

  it('disables Export Video buttons for a run without a candidate NPZ', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await user.click(await screen.findByRole('button', { name: /opensim/i }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /export video \(gif\)/i })).toBeDisabled();
      expect(screen.getByRole('button', { name: /export video \(mp4\)/i })).toBeDisabled();
    });
  });

  it('disables the parity button when unavailable and fetches it when available', async () => {
    const user = userEvent.setup();
    const { container } = render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    const parityButton = screen.getByRole('button', { name: /view parity report/i });
    expect(parityButton).toBeEnabled();
    await user.click(parityButton);
    await waitFor(() => {
      expect(fetchParityReportMock).toHaveBeenCalledWith('aaa111');
    });
    await waitFor(() => {
      expect(container.textContent).toContain('matched-swing-parity-report-v1');
    });

    await user.click(screen.getByRole('button', { name: /opensim/i }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /view parity report/i })).toBeDisabled();
    });
  });

  it('requests the ledger ranked so the best candidate is auto-selected', async () => {
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByText('Matched Swing Results');
    await waitFor(() => {
      expect(fetchMatchedSwingLedger).toHaveBeenCalledWith(
        expect.objectContaining({ ranked: true }),
      );
    });
  });

  it('refetches the ledger when the Drive Mode or Profile filter changes', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    vi.mocked(fetchMatchedSwingLedger).mockClear();

    const driveModeSelect = screen.getAllByRole('combobox')[4];
    await user.selectOptions(driveModeSelect, 'torque_driven');
    await waitFor(() => {
      expect(fetchMatchedSwingLedger).toHaveBeenCalledWith(
        expect.objectContaining({ ranked: true, driveMode: 'torque_driven' }),
      );
    });

    vi.mocked(fetchMatchedSwingLedger).mockClear();
    const profileSelect = screen.getAllByRole('combobox')[5];
    await user.selectOptions(profileSelect, 'dynamic');
    await waitFor(() => {
      expect(fetchMatchedSwingLedger).toHaveBeenCalledWith(
        expect.objectContaining({ ranked: true, profile: 'dynamic' }),
      );
    });
  });

  it('prefers qualification_note over reason in the rejection/qualification text', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await user.click(await screen.findByRole('button', { name: /opensim/i }));
    await waitFor(() => {
      expect(screen.getByText('preferred qualification note')).toBeInTheDocument();
      expect(screen.queryByText('unique_rejection_marker')).not.toBeInTheDocument();
    });
  });

  it('launches native viewer successfully for a run with candidate NPZ', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    const launchBtn = screen.getByRole('button', { name: /open native viewer/i });
    expect(launchBtn).toBeEnabled();
    await user.click(launchBtn);

    await waitFor(() => {
      expect(launchMatchedSwingNativeViewerMock).toHaveBeenCalledWith('aaa111');
    });
    expect(
      await screen.findByText('Native viewer meshcat launched successfully'),
    ).toBeInTheDocument();
  });

  it('displays error when launching native viewer fails', async () => {
    launchMatchedSwingNativeViewerMock.mockRejectedValueOnce(
      new Error('Native viewer meshcat unavailable'),
    );
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await screen.findByRole('button', { name: /drake/i });
    const launchBtn = screen.getByRole('button', { name: /open native viewer/i });
    await user.click(launchBtn);

    expect(
      await screen.findByText('Native viewer meshcat unavailable'),
    ).toBeInTheDocument();
  });

  it('disables Open Native Viewer button for a run without candidate NPZ', async () => {
    const user = userEvent.setup();
    render(
      <MemoryRouter>
        <MatchedSwingsPage />
      </MemoryRouter>,
    );

    await user.click(await screen.findByRole('button', { name: /opensim/i }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /open native viewer/i })).toBeDisabled();
    });
  });
});

