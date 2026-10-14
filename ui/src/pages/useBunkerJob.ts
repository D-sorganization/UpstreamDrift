/**
 * One BunkerShot3D workbench job at a time (#9545, slice 5).
 *
 * Submits a solve to the job routes, polls its status on the bounded,
 * backed-off `JOB_POLL` schedule and settles on the first terminal state. A
 * newer submission, a cancel, a terminal state and unmounting each abort the
 * poll, and nothing that answers an older submission is applied. A completed
 * result is shown only when its job id, kind and input digest match the
 * submission (`freshJobRecord`); anything else settles as `stale`.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  cancelBunkerJob,
  freshJobRecord,
  getBunkerJob,
  isTerminalJob,
  pollBunkerJob,
  submitBunkerJob,
  ticketOf,
  type BunkerJobKind,
  type BunkerJobRequests,
  type BunkerJobStatusV1,
  type BunkerJobTicket,
} from '@/api/bunkerWorkbench';

export type BunkerJobView =
  | { phase: 'idle' }
  | { phase: 'submitting' }
  | { phase: 'running'; status: BunkerJobStatusV1 }
  | { phase: 'completed'; kind: BunkerJobKind; record: Record<string, unknown> }
  | { phase: 'cancelled' }
  | { phase: 'failed'; message: string }
  | { phase: 'stale' };

interface JobRun {
  controller: AbortController;
  ticket: BunkerJobTicket | null;
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * The view a status gives the submission `ticket`, or `null` to ignore it.
 *
 * A live status for another job is ignored; a terminal one, like a completed
 * result that does not answer the ticket, settles as `stale`.
 */
export function viewOfStatus(
  status: BunkerJobStatusV1,
  ticket: BunkerJobTicket,
): BunkerJobView | null {
  const ours = status.job_id === ticket.jobId;
  if (!isTerminalJob(status)) {
    return ours ? { phase: 'running', status } : null;
  }
  if (!ours) return { phase: 'stale' };
  switch (status.status) {
    case 'completed': {
      const record = freshJobRecord(status, ticket);
      return record ? { phase: 'completed', kind: ticket.kind, record } : { phase: 'stale' };
    }
    case 'failed':
      return { phase: 'failed', message: `Job failed: ${status.error ?? 'no reason was given'}` };
    default:
      return { phase: 'cancelled' };
  }
}

export function useBunkerJob() {
  const [view, setView] = useState<BunkerJobView>({ phase: 'idle' });
  const [notice, setNotice] = useState<string | null>(null);
  const current = useRef<JobRun | null>(null);

  const stop = useCallback(() => {
    current.current?.controller.abort();
    current.current = null;
  }, []);

  useEffect(() => stop, [stop]);

  const apply = useCallback(
    (owner: JobRun, status: BunkerJobStatusV1) => {
      if (current.current !== owner || owner.ticket === null) return;
      const next = viewOfStatus(status, owner.ticket);
      if (next === null) return;
      setView(next);
      if (isTerminalJob(status)) stop();
    },
    [stop],
  );

  async function submit<K extends BunkerJobKind>(kind: K, build: () => BunkerJobRequests[K]) {
    stop();
    const owner: JobRun = { controller: new AbortController(), ticket: null };
    current.current = owner;
    setNotice(null);
    setView({ phase: 'submitting' });
    try {
      const accepted = await submitBunkerJob(kind, build(), owner.controller.signal);
      if (current.current !== owner) return;
      owner.ticket = { ...ticketOf(accepted), kind };
      apply(owner, accepted);
      if (current.current !== owner) return;
      await pollBunkerJob(owner.ticket.jobId, {
        signal: owner.controller.signal,
        onStatus: (status) => apply(owner, status),
        fetchStatus: getBunkerJob,
      });
    } catch (err) {
      if (current.current !== owner) return;
      current.current = null;
      setView({ phase: 'failed', message: errorText(err) });
    }
  }

  async function cancel() {
    const owner = current.current;
    if (owner?.ticket == null) return;
    try {
      apply(owner, await cancelBunkerJob(owner.ticket.jobId));
    } catch (err) {
      if (current.current === owner) setNotice(`Cancel failed: ${errorText(err)}`);
    }
  }

  const busy = view.phase === 'submitting' || view.phase === 'running';
  return { view, notice, busy, submit, cancel };
}
