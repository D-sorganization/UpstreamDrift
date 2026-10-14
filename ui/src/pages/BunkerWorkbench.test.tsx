/**
 * Tests for the BunkerShot3D workbench web page (#9545, slices 3 and 5).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BunkerWorkbenchPage } from './BunkerWorkbench';
import {
  JOB_POLL,
  type BunkerJobStatusV1,
  type BunkerWorkbenchComparisonV1,
  type BunkerWorkbenchEvaluationV1,
} from '@/api/bunkerWorkbench';

function evaluation(name: string, carry: number | null): BunkerWorkbenchEvaluationV1 {
  return {
    schema_version: 'bunker-workbench-evaluation/1',
    conventions: {},
    inputs: { design: { name } },
    tier: 'F0',
    tier_model: 'dynamic RFT',
    verdict: {
      status: 'extrapolated',
      headline: `${name}: outside the calibrated envelope`,
      caveats: [],
      reasons: ['firmness outside the calibrated range'],
      reason_count: 1,
    },
    stamp: 'F0 dynamic RFT | EXTRAPOLATED',
    refused: false,
    shot: {
      peak_force_n: 120,
      impulse_n_s: 1.5,
      entry_speed_mps: 20,
      exit_speed_mps: 12,
      max_depth_m: 0.02,
      contact_duration_s: 0.01,
      peak_inertial_fraction: 0.3,
    },
    delivered: {
      effective_loft_deg: 60,
      effective_marketed_bounce_deg: 12,
      presentation_bounce_deg: 14,
      aim_offset_deg: 1,
    },
    carry:
      carry === null
        ? null
        : { carry_m: carry, verdict_status: 'extrapolated', band_m: null, band_reasons: [] },
    unavailable: [],
    sources: {
      model: 'src.tools.bunker_shot_gui.model.WorkbenchModel',
      sand: { firmness: { basis: 'preset', source: 'medium sand preset' } },
      carry_band: null,
    },
    objective: null,
    predictive: false,
    digest: 'a'.repeat(64),
  };
}

function comparison(
  overrides: Partial<BunkerWorkbenchComparisonV1> = {},
): BunkerWorkbenchComparisonV1 {
  return {
    schema_version: 'bunker-workbench-comparison/1',
    conventions: {},
    left: evaluation('Design A', 9.5),
    right: evaluation('Design B', 8.1),
    objective: { use: 'exploratory', target_carry_m: 9, tolerance_fraction: 0.15 },
    objective_standing: {
      disposition: 'unavailable-uncalibrated',
      reasons: ['no sand-to-ball transfer qualification ships'],
      predictive: false,
    },
    ranking: {
      available: true,
      unavailable_reason: null,
      shared_points: 25,
      banded_verdict: 'separated',
      separation_m: 1.2,
      separated: true,
    },
    predictive: false,
    winner: null,
    exploratory_leader: 'Design A',
    verdict_statement: 'Design A leads on an exploratory objective',
    digest: 'b'.repeat(64),
    ...overrides,
  };
}

const JOB_ID = '11111111-1111-4111-8111-111111111111';
const INPUT_DIGEST = 'd'.repeat(64);

function job(overrides: Partial<BunkerJobStatusV1> = {}): BunkerJobStatusV1 {
  return {
    job_id: JOB_ID,
    kind: 'compare',
    status: 'running',
    progress_percent: 0,
    shots_planned: 52,
    input_digest: INPUT_DIGEST,
    worker_active: true,
    error: null,
    result: null,
    ...overrides,
  };
}

/** A completed job whose result answers the submission. */
function done(record: object, kind: BunkerJobStatusV1['kind'] = 'compare'): BunkerJobStatusV1 {
  return job({
    kind,
    status: 'completed',
    progress_percent: 100,
    worker_active: false,
    result: {
      job_id: JOB_ID,
      input_digest: INPUT_DIGEST,
      record: record as Record<string, unknown>,
    },
  });
}

const { submitMock, getMock, cancelMock } = vi.hoisted(() => ({
  submitMock: vi.fn(),
  getMock: vi.fn(),
  cancelMock: vi.fn(),
}));

vi.mock('@/api/bunkerWorkbench', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api/bunkerWorkbench')>();
  return {
    ...actual,
    submitBunkerJob: submitMock,
    getBunkerJob: getMock,
    cancelBunkerJob: cancelMock,
  };
});

beforeEach(() => {
  submitMock.mockReset();
  getMock.mockReset();
  cancelMock.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('BunkerWorkbenchPage results', () => {
  it('solves design A as a job and shows its tier, verdict, stamp and sources', async () => {
    submitMock.mockResolvedValue(done(evaluation('Design A', 9.5), 'evaluate'));
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    const loft = screen.getByLabelText('Design A loft (deg)');
    await user.clear(loft);
    await user.type(loft, '58');
    await user.click(screen.getByRole('button', { name: 'Solve design A' }));

    expect(await screen.findByText('F0 dynamic RFT | EXTRAPOLATED')).toBeInTheDocument();
    expect(screen.getByText('Design A: outside the calibrated envelope')).toBeInTheDocument();
    expect(screen.getByText(/firmness: preset/)).toBeInTheDocument();
    expect(screen.getByText('9.50 m')).toBeInTheDocument();
    const [kind, request] = submitMock.mock.calls[0];
    expect(kind).toBe('evaluate');
    expect(request.handedness).toBe('right');
    expect(request.design).toMatchObject({ name: 'Design A', loft_deg: 58 });
    expect(request.objective).toMatchObject({ use: 'exploratory' });
  });

  it('shows an unqualified A/B lead as exploratory, never as a winner', async () => {
    submitMock.mockResolvedValue(done(comparison()));
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByText('unavailable-uncalibrated')).toBeInTheDocument();
    expect(submitMock.mock.calls[0][0]).toBe('compare');
    expect(screen.getByText('no sand-to-ball transfer qualification ships')).toBeInTheDocument();
    expect(screen.getByTestId('bunker-winner')).toHaveTextContent('Not ranked');
    expect(screen.getByTestId('bunker-exploratory-leader')).toHaveTextContent('Design A');
    expect(screen.getAllByText('F0 dynamic RFT | EXTRAPOLATED')).toHaveLength(2);
  });

  it('demotes a winner sent on a non-predictive standing', async () => {
    submitMock.mockResolvedValue(done(comparison({ winner: 'Design B', exploratory_leader: null })));
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByTestId('bunker-winner')).toHaveTextContent('Not ranked');
    expect(screen.getByTestId('bunker-exploratory-leader')).toHaveTextContent('Design B');
  });

  it('names the winner only on a supported, predictive standing', async () => {
    submitMock.mockResolvedValue(
      done(
        comparison({
          predictive: true,
          winner: 'Design B',
          exploratory_leader: null,
          objective_standing: { disposition: 'supported', reasons: [], predictive: true },
        }),
      ),
    );
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByTestId('bunker-winner')).toHaveTextContent('Design B');
    expect(screen.queryByTestId('bunker-exploratory-leader')).not.toBeInTheDocument();
  });

  it('refuses a non-numeric field instead of sending the preset value', async () => {
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    const width = screen.getByLabelText('Design B sole width (mm)');
    await user.clear(width);
    await user.type(width, 'wide');
    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Design B sole width (mm) must be a number',
    );
    expect(submitMock).not.toHaveBeenCalled();
  });

  it('shows the route refusal instead of a result', async () => {
    submitMock.mockRejectedValue(new Error('the two designs must have different names'));
    const user = userEvent.setup();
    render(<BunkerWorkbenchPage />);

    await user.click(screen.getByRole('button', { name: 'Compare A/B' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'the two designs must have different names',
    );
    expect(screen.queryByTestId('bunker-winner')).not.toBeInTheDocument();
  });
});

/** Click Compare A/B and let the 202 settle, under fake timers. */
async function submitCompare(): Promise<void> {
  await act(async () => {
    fireEvent.click(screen.getByRole('button', { name: 'Compare A/B' }));
  });
}

async function advance(ms: number): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

describe('BunkerWorkbenchPage jobs', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('shows labelled progress while polling and stops once the job completes', async () => {
    submitMock.mockResolvedValue(job({ status: 'pending' }));
    getMock
      .mockResolvedValueOnce(job({ progress_percent: 40 }))
      .mockResolvedValueOnce(done(comparison()));
    render(<BunkerWorkbenchPage />);

    await submitCompare();
    const progress = screen.getByRole('progressbar', { name: 'Bunker job progress' });
    expect(progress).toHaveAttribute('value', '0');

    await advance(JOB_POLL.initialMs);
    expect(getMock).toHaveBeenCalledWith(JOB_ID, expect.any(AbortSignal));
    expect(screen.getByRole('progressbar', { name: 'Bunker job progress' })).toHaveAttribute(
      'value',
      '40',
    );
    expect(screen.getByText('40% of 52 shots')).toBeInTheDocument();

    await advance(JOB_POLL.initialMs * JOB_POLL.factor);
    expect(screen.getByTestId('bunker-winner')).toHaveTextContent('Not ranked');
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Cancel job' })).not.toBeInTheDocument();

    await advance(10 * JOB_POLL.maxMs);
    expect(getMock).toHaveBeenCalledTimes(2);
  });

  it('cancels the running job from a keyboard-reachable button', async () => {
    submitMock.mockResolvedValue(job({ progress_percent: 10 }));
    getMock.mockResolvedValue(job({ progress_percent: 10 }));
    cancelMock.mockResolvedValue(job({ status: 'cancelled', worker_active: false }));
    render(<BunkerWorkbenchPage />);

    await submitCompare();
    const cancel = screen.getByRole('button', { name: 'Cancel job' });
    cancel.focus();
    expect(cancel).toHaveFocus();
    await act(async () => {
      fireEvent.click(cancel);
    });

    expect(cancelMock).toHaveBeenCalledWith(JOB_ID);
    expect(screen.getByRole('status')).toHaveTextContent('Job cancelled');
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.queryByTestId('bunker-winner')).not.toBeInTheDocument();
    await advance(10 * JOB_POLL.maxMs);
    expect(getMock).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Compare A/B' })).toBeEnabled();
  });

  it('shows a failed job with its reason and no result', async () => {
    submitMock.mockResolvedValue(job());
    getMock.mockResolvedValue(
      job({ status: 'failed', worker_active: false, error: 'ValueError: sole unconstructible' }),
    );
    render(<BunkerWorkbenchPage />);

    await submitCompare();
    await advance(JOB_POLL.initialMs);

    expect(screen.getByRole('alert')).toHaveTextContent(
      'Job failed: ValueError: sole unconstructible',
    );
    expect(screen.queryByTestId('bunker-winner')).not.toBeInTheDocument();
  });

  it('does not show a result for another input digest as fresh', async () => {
    submitMock.mockResolvedValue(job());
    getMock.mockResolvedValue(
      job({
        status: 'completed',
        progress_percent: 100,
        result: { job_id: JOB_ID, input_digest: 'e'.repeat(64), record: { ...comparison() } },
      }),
    );
    render(<BunkerWorkbenchPage />);

    await submitCompare();
    await advance(JOB_POLL.initialMs);

    expect(screen.getByRole('alert')).toHaveTextContent('stale result');
    expect(screen.queryByTestId('bunker-winner')).not.toBeInTheDocument();
  });

  it('does not show a result naming another job as fresh', async () => {
    const other = '22222222-2222-4222-8222-222222222222';
    submitMock.mockResolvedValue(job());
    getMock.mockResolvedValue(
      job({
        status: 'completed',
        progress_percent: 100,
        result: { job_id: other, input_digest: INPUT_DIGEST, record: { ...comparison() } },
      }),
    );
    render(<BunkerWorkbenchPage />);

    await submitCompare();
    await advance(JOB_POLL.initialMs);

    expect(screen.getByRole('alert')).toHaveTextContent('stale result');
    expect(screen.queryByTestId('bunker-winner')).not.toBeInTheDocument();
  });

  it('stops polling when the page unmounts', async () => {
    submitMock.mockResolvedValue(job());
    getMock.mockResolvedValue(job({ progress_percent: 5 }));
    const { unmount } = render(<BunkerWorkbenchPage />);

    await submitCompare();
    await advance(JOB_POLL.initialMs);
    expect(getMock).toHaveBeenCalledTimes(1);
    unmount();

    await advance(10 * JOB_POLL.maxMs);
    expect(getMock).toHaveBeenCalledTimes(1);
  });
});
