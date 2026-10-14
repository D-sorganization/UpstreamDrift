/**
 * BunkerShot3D designer workbench client (#9545, slices 3 and 5).
 *
 * Typed client for the job routes in `src/api/routes/bunker_workbench_jobs.py`:
 * `POST /tools/bunker-workbench/v1/jobs/evaluate` (one bounded nominal F0
 * solve) and `.../jobs/compare` (the A/B comparison with the model's
 * carry-objective standing) answer at once with a job id and input digest;
 * `GET .../jobs/{job_id}` reports status and per-shot progress, and
 * `POST .../jobs/{job_id}/cancel` stops the job before its next shot. The
 * route adds no physics; neither does this client. It only refuses to promote
 * an unqualified lead to a ranking, or a result for another submission to a
 * fresh one.
 */

import { apiFetch } from './fetch';
import type {
  BunkerJobStatusV1,
  BunkerWorkbenchCompareRequestV1,
  BunkerWorkbenchComparisonV1,
  BunkerWorkbenchRequestV1,
} from './generated/types';

export type {
  BunkerDesignV1,
  BunkerJobStatusV1,
  BunkerWorkbenchCompareRequestV1,
  BunkerWorkbenchComparisonV1,
  BunkerWorkbenchEvaluationV1,
  BunkerWorkbenchRequestV1,
} from './generated/types';

/** Versioned route prefix, mounted under the legacy `/api` prefix. */
export const BUNKER_WORKBENCH_PATH = '/api/tools/bunker-workbench/v1';

/** A job id is the UUID the route minted; anything else is refused unsent. */
const JOB_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** A submit, status read or cancel answers at once; the solve runs in the job. */
const JOB_REQUEST_TIMEOUT_MS = 15_000;

/**
 * Status polling schedule: the first read after `initialMs`, each later wait
 * `factor` times the previous one up to `maxMs`, and a give-up after
 * `maxConsecutiveErrors` failed reads in a row.
 */
export const JOB_POLL = {
  initialMs: 250,
  factor: 2,
  maxMs: 2_000,
  maxConsecutiveErrors: 3,
} as const;

export type BunkerJobKind = BunkerJobStatusV1['kind'];

/** The request each job kind accepts. */
export interface BunkerJobRequests {
  evaluate: BunkerWorkbenchRequestV1;
  compare: BunkerWorkbenchCompareRequestV1;
}

/** What the answer to a submission must name: its job, kind and input digest. */
export interface BunkerJobTicket {
  jobId: string;
  kind: BunkerJobKind;
  inputDigest: string;
}

const TERMINAL_STATES: ReadonlySet<BunkerJobStatusV1['status']> = new Set([
  'completed',
  'failed',
  'cancelled',
]);

function jobPath(jobId: string): string {
  if (!JOB_ID.test(jobId)) {
    throw new Error(`"${jobId}" is not a bunker job id`);
  }
  return `${BUNKER_WORKBENCH_PATH}/jobs/${jobId}`;
}

/** Submit a bounded solve as a job; a 422 refusal rejects with its message. */
export function submitBunkerJob<K extends BunkerJobKind>(
  kind: K,
  request: BunkerJobRequests[K],
  signal?: AbortSignal,
): Promise<BunkerJobStatusV1> {
  return apiFetch<BunkerJobStatusV1>(`${BUNKER_WORKBENCH_PATH}/jobs/${kind}`, {
    method: 'POST',
    body: JSON.stringify(request),
    timeoutMs: JOB_REQUEST_TIMEOUT_MS,
    signal,
  });
}

/** The job's status and progress; its result only once `completed`. */
export async function getBunkerJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<BunkerJobStatusV1> {
  return apiFetch<BunkerJobStatusV1>(jobPath(jobId), {
    timeoutMs: JOB_REQUEST_TIMEOUT_MS,
    signal,
  });
}

/** Cancel a pending or running job; it stops before its next shot. */
export async function cancelBunkerJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<BunkerJobStatusV1> {
  return apiFetch<BunkerJobStatusV1>(`${jobPath(jobId)}/cancel`, {
    method: 'POST',
    timeoutMs: JOB_REQUEST_TIMEOUT_MS,
    signal,
  });
}

/** Whether the job has settled: completed, failed or cancelled. */
export function isTerminalJob(status: BunkerJobStatusV1): boolean {
  return TERMINAL_STATES.has(status.status);
}

/** The wait before the next status read: backed off, never above the cap. */
export function nextPollDelay(previousMs: number): number {
  return Math.min(previousMs * JOB_POLL.factor, JOB_POLL.maxMs);
}

/** The identity a submission's answer must carry, taken from its 202 status. */
export function ticketOf(accepted: BunkerJobStatusV1): BunkerJobTicket {
  return { jobId: accepted.job_id, kind: accepted.kind, inputDigest: accepted.input_digest };
}

/**
 * The record of a completed job, only when it answers `ticket`.
 *
 * A status or result naming another job, kind or input digest is stale and
 * yields `null`; so does every status that is not `completed`.
 */
export function freshJobRecord(
  status: BunkerJobStatusV1,
  ticket: BunkerJobTicket,
): Record<string, unknown> | null {
  const result = status.result;
  if (
    status.status !== 'completed' ||
    status.job_id !== ticket.jobId ||
    status.kind !== ticket.kind ||
    status.input_digest !== ticket.inputDigest ||
    !result ||
    result.job_id !== ticket.jobId ||
    result.input_digest !== ticket.inputDigest
  ) {
    return null;
  }
  return result.record;
}

function abortError(): DOMException {
  return new DOMException('The bunker job poll was stopped.', 'AbortError');
}

function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) {
      reject(abortError());
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

export interface PollBunkerJobOptions {
  /** Stops the poll; the promise then rejects with an `AbortError`. */
  signal: AbortSignal;
  /** Called with every status read, terminal or not. */
  onStatus: (status: BunkerJobStatusV1) => void;
  /** The status read; defaults to {@link getBunkerJob}. */
  fetchStatus?: (jobId: string, signal: AbortSignal) => Promise<BunkerJobStatusV1>;
}

/**
 * Poll a job on the {@link JOB_POLL} schedule until it settles.
 *
 * @returns The first terminal status.
 * @throws AbortError when `signal` aborts; the last read error after
 *   `JOB_POLL.maxConsecutiveErrors` failed reads in a row.
 */
export async function pollBunkerJob(
  jobId: string,
  { signal, onStatus, fetchStatus = getBunkerJob }: PollBunkerJobOptions,
): Promise<BunkerJobStatusV1> {
  let delay: number = JOB_POLL.initialMs;
  let failures = 0;
  for (;;) {
    await wait(delay, signal);
    delay = nextPollDelay(delay);
    let status: BunkerJobStatusV1;
    try {
      status = await fetchStatus(jobId, signal);
    } catch (err) {
      if (signal.aborted) throw abortError();
      failures += 1;
      if (failures >= JOB_POLL.maxConsecutiveErrors) throw err;
      continue;
    }
    if (signal.aborted) throw abortError();
    failures = 0;
    onStatus(status);
    if (isTerminalJob(status)) return status;
  }
}

function isPredictive(comparison: BunkerWorkbenchComparisonV1): boolean {
  return comparison.predictive && comparison.objective_standing.predictive;
}

/**
 * The design a client may present as ranked first, or `null`.
 *
 * Only a predictive standing ranks (#9239). The route already sends `winner`
 * as `null` otherwise; this guard holds even if a record says otherwise.
 */
export function rankedWinner(comparison: BunkerWorkbenchComparisonV1): string | null {
  return isPredictive(comparison) ? comparison.winner : null;
}

/**
 * The design the bands favour on an unqualified objective, or `null`.
 *
 * A `winner` on a non-predictive standing is demoted here rather than shown.
 */
export function exploratoryLeader(
  comparison: BunkerWorkbenchComparisonV1,
): string | null {
  if (isPredictive(comparison)) return null;
  return comparison.exploratory_leader ?? comparison.winner;
}
