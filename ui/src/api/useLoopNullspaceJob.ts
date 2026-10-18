/**
 * Submit, poll and cancel one loop-nullspace job (NSX-44, #12576).
 *
 * Polls with backoff until the job settles; the latest status (with its partial
 * or final body) is exposed as is. A cache hit arrives already completed.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  LOOP_NULLSPACE_POLL,
  cancelLoopNullspaceJob,
  getLoopNullspaceJob,
  isTerminalLoopNullspaceJob,
  nextLoopNullspaceDelay,
  submitLoopNullspaceJob,
  type LoopNullspaceJobStatusV1,
  type LoopNullspaceJobRequest,
} from './loopNullSpace';

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

export function useLoopNullspaceJob() {
  const [job, setJob] = useState<LoopNullspaceJobStatusV1 | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const run = useRef(0);

  const stopTimer = useCallback(() => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
  }, []);

  useEffect(() => stopTimer, [stopTimer]);

  const poll = useCallback((jobId: string, token: number, delay: number) => {
    timer.current = setTimeout(async () => {
      try {
        const next = await getLoopNullspaceJob(jobId);
        if (token !== run.current) return;
        setJob(next);
        if (!isTerminalLoopNullspaceJob(next)) poll(jobId, token, nextLoopNullspaceDelay(delay));
      } catch (e) {
        if (token !== run.current) return;
        setError(message(e));
      }
    }, delay);
  }, []);

  const start = useCallback(
    async (request: LoopNullspaceJobRequest) => {
      stopTimer();
      const token = ++run.current;
      setError(null);
      setJob(null);
      setBusy(true);
      try {
        const accepted = await submitLoopNullspaceJob(request);
        if (token !== run.current) return;
        setJob(accepted);
        if (!isTerminalLoopNullspaceJob(accepted)) {
          poll(accepted.job_id, token, LOOP_NULLSPACE_POLL.initialMs);
        }
      } catch (e) {
        if (token === run.current) setError(message(e));
      } finally {
        if (token === run.current) setBusy(false);
      }
    },
    [poll, stopTimer],
  );

  const cancel = useCallback(async () => {
    if (!job) return;
    stopTimer();
    try {
      const cancelled = await cancelLoopNullspaceJob(job.job_id);
      run.current += 1;
      setJob(cancelled);
    } catch (e) {
      setError(message(e));
    }
  }, [job, stopTimer]);

  return { job, error, busy, start, cancel };
}
