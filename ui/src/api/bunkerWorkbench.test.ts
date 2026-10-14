/**
 * Tests for the BunkerShot3D workbench client (#9545, slices 3 and 5).
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  BUNKER_WORKBENCH_PATH,
  JOB_POLL,
  cancelBunkerJob,
  exploratoryLeader,
  freshJobRecord,
  getBunkerJob,
  isTerminalJob,
  nextPollDelay,
  pollBunkerJob,
  rankedWinner,
  submitBunkerJob,
  ticketOf,
  type BunkerJobStatusV1,
} from './bunkerWorkbench';
import type { BunkerWorkbenchComparisonV1 } from './generated/types';

function response(ok: boolean, body: unknown, status = 200): Response {
  return {
    ok,
    status,
    statusText: ok ? 'OK' : 'Unprocessable Content',
    json: () => Promise.resolve(body),
  } as unknown as Response;
}

function comparison(
  overrides: Partial<BunkerWorkbenchComparisonV1>,
): BunkerWorkbenchComparisonV1 {
  return {
    predictive: false,
    winner: null,
    exploratory_leader: null,
    objective_standing: { disposition: 'unavailable-uncalibrated', reasons: [], predictive: false },
    ...overrides,
  } as BunkerWorkbenchComparisonV1;
}

const JOB_ID = '11111111-1111-4111-8111-111111111111';

function job(overrides: Partial<BunkerJobStatusV1> = {}): BunkerJobStatusV1 {
  return {
    job_id: JOB_ID,
    kind: 'compare',
    status: 'running',
    progress_percent: 0,
    shots_planned: 52,
    input_digest: 'd'.repeat(64),
    worker_active: true,
    error: null,
    result: null,
    ...overrides,
  };
}

function completed(overrides: Partial<BunkerJobStatusV1> = {}): BunkerJobStatusV1 {
  return job({
    status: 'completed',
    progress_percent: 100,
    worker_active: false,
    result: { job_id: JOB_ID, input_digest: 'd'.repeat(64), record: { digest: 'r' } },
    ...overrides,
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('bunker job requests', () => {
  it('submits an evaluation job to the versioned jobs route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response(true, job({ kind: 'evaluate' }), 202));
    vi.stubGlobal('fetch', fetchMock);

    const accepted = await submitBunkerJob('evaluate', { handedness: 'right', design: { name: 'A' } });

    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toContain(`${BUNKER_WORKBENCH_PATH}/jobs/evaluate`);
    expect(init.method).toBe('POST');
    expect(JSON.parse(init.body as string)).toEqual({ handedness: 'right', design: { name: 'A' } });
    expect(accepted.job_id).toBe(JOB_ID);
  });

  it('submits a comparison job, reads its status and cancels it by id', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response(true, job()));
    vi.stubGlobal('fetch', fetchMock);

    await submitBunkerJob('compare', { handedness: 'right', left: { name: 'A' }, right: { name: 'B' } });
    await getBunkerJob(JOB_ID);
    await cancelBunkerJob(JOB_ID);

    const calls = fetchMock.mock.calls as [string, RequestInit | undefined][];
    expect(calls[0][0]).toContain(`${BUNKER_WORKBENCH_PATH}/jobs/compare`);
    expect(calls[1][0]).toContain(`${BUNKER_WORKBENCH_PATH}/jobs/${JOB_ID}`);
    expect(calls[1][1]?.method ?? 'GET').toBe('GET');
    expect(calls[2][0]).toContain(`${BUNKER_WORKBENCH_PATH}/jobs/${JOB_ID}/cancel`);
    expect(calls[2][1]?.method).toBe('POST');
  });

  it('refuses a job id that is not a UUID before any request', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);

    await expect(getBunkerJob('../evaluate')).rejects.toThrow('not a bunker job id');
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('surfaces the message of a structured 422 refusal', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        response(
          false,
          { detail: { code: 'design_names_not_distinct', message: 'names must differ' } },
          422,
        ),
      ),
    );
    await expect(
      submitBunkerJob('compare', { handedness: 'right', left: { name: 'A' }, right: { name: 'A' } }),
    ).rejects.toThrow('names must differ');
  });
});

describe('job polling', () => {
  it('backs off from the initial interval up to the cap', () => {
    expect(nextPollDelay(JOB_POLL.initialMs)).toBe(JOB_POLL.initialMs * JOB_POLL.factor);
    expect(nextPollDelay(JOB_POLL.maxMs)).toBe(JOB_POLL.maxMs);
    expect(nextPollDelay(JOB_POLL.maxMs * 10)).toBe(JOB_POLL.maxMs);
  });

  it('knows the terminal states', () => {
    expect(isTerminalJob(job({ status: 'pending' }))).toBe(false);
    expect(isTerminalJob(job({ status: 'running' }))).toBe(false);
    for (const status of ['completed', 'failed', 'cancelled'] as const) {
      expect(isTerminalJob(job({ status }))).toBe(true);
    }
  });

  it('polls with backoff and stops at the first terminal state', async () => {
    vi.useFakeTimers();
    const fetchStatus = vi
      .fn()
      .mockResolvedValueOnce(job({ progress_percent: 10 }))
      .mockResolvedValueOnce(job({ progress_percent: 50 }))
      .mockResolvedValueOnce(completed());
    const seen: number[] = [];
    const done = pollBunkerJob(JOB_ID, {
      signal: new AbortController().signal,
      onStatus: (status) => seen.push(status.progress_percent),
      fetchStatus,
    });

    await vi.advanceTimersByTimeAsync(JOB_POLL.initialMs - 1);
    expect(fetchStatus).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(JOB_POLL.initialMs * JOB_POLL.factor);
    expect(fetchStatus).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(JOB_POLL.maxMs);

    await expect(done).resolves.toMatchObject({ status: 'completed' });
    expect(seen).toEqual([10, 50, 100]);
    await vi.advanceTimersByTimeAsync(10 * JOB_POLL.maxMs);
    expect(fetchStatus).toHaveBeenCalledTimes(3);
  });

  it('stops polling when its signal aborts', async () => {
    vi.useFakeTimers();
    const fetchStatus = vi.fn().mockResolvedValue(job());
    const controller = new AbortController();
    const done = pollBunkerJob(JOB_ID, { signal: controller.signal, onStatus: () => {}, fetchStatus });
    const settled = expect(done).rejects.toMatchObject({ name: 'AbortError' });

    await vi.advanceTimersByTimeAsync(JOB_POLL.initialMs);
    controller.abort();
    await settled;
    await vi.advanceTimersByTimeAsync(10 * JOB_POLL.maxMs);
    expect(fetchStatus).toHaveBeenCalledTimes(1);
  });

  it('gives up after a bounded run of failed status reads', async () => {
    vi.useFakeTimers();
    const fetchStatus = vi.fn().mockRejectedValue(new Error('connection refused'));
    const done = pollBunkerJob(JOB_ID, {
      signal: new AbortController().signal,
      onStatus: () => {},
      fetchStatus,
    });
    const settled = expect(done).rejects.toThrow('connection refused');

    await vi.advanceTimersByTimeAsync(JOB_POLL.maxConsecutiveErrors * JOB_POLL.maxMs);
    await settled;
    expect(fetchStatus).toHaveBeenCalledTimes(JOB_POLL.maxConsecutiveErrors);
  });
});

describe('stale-result guard', () => {
  const ticket = ticketOf(job({ status: 'pending' }));

  it('returns the record of the job it submitted', () => {
    expect(freshJobRecord(completed(), ticket)).toEqual({ digest: 'r' });
  });

  it('drops a result for another input digest', () => {
    const stale = completed({
      result: { job_id: JOB_ID, input_digest: 'e'.repeat(64), record: { digest: 'r' } },
    });
    expect(freshJobRecord(stale, ticket)).toBeNull();
  });

  it('drops a result naming another job, or a status for another job', () => {
    const otherId = '22222222-2222-4222-8222-222222222222';
    const foreignResult = completed({
      result: { job_id: otherId, input_digest: 'd'.repeat(64), record: { digest: 'r' } },
    });
    expect(freshJobRecord(foreignResult, ticket)).toBeNull();
    expect(freshJobRecord(completed({ job_id: otherId }), ticket)).toBeNull();
    expect(freshJobRecord(completed({ kind: 'evaluate' }), ticket)).toBeNull();
  });

  it('has no record for a job that did not complete', () => {
    expect(freshJobRecord(job({ status: 'cancelled' }), ticket)).toBeNull();
  });
});

describe('ranking guard (#9239)', () => {
  it('names a winner only on a predictive standing', () => {
    const supported = comparison({
      predictive: true,
      winner: 'A',
      objective_standing: { disposition: 'supported', reasons: [], predictive: true },
    });
    expect(rankedWinner(supported)).toBe('A');
    expect(exploratoryLeader(supported)).toBeNull();
  });

  it('never shows a winner on an unqualified objective, even if one is sent', () => {
    const unqualified = comparison({ predictive: false, winner: 'A' });
    expect(rankedWinner(unqualified)).toBeNull();
    expect(exploratoryLeader(unqualified)).toBe('A');
  });

  it('reports the exploratory leader as given', () => {
    expect(exploratoryLeader(comparison({ exploratory_leader: 'B' }))).toBe('B');
    expect(rankedWinner(comparison({ exploratory_leader: 'B' }))).toBeNull();
  });
});
