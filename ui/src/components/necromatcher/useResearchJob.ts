import { useEffect, useRef, useState } from 'react';
import type { ResearchRun } from '@/api/necromatcher';

type JobProps<R extends ResearchRun, P> = {
  fit: string; initialRunId?: string; onRun?: (run: string) => void; onCompleted?: () => void;
  api: {submit: (fit: string, payload: P) => Promise<R>; view: (run: string) => Promise<R>; cancel: (run: string) => Promise<R>};
};
const errorText = (error: unknown) => error instanceof Error ? error.message : 'The research job could not be reached.';

export function useResearchJob<R extends ResearchRun, P>({fit, onCompleted, initialRunId, onRun, api}: JobProps<R, P>) {
  const [run, setRun] = useState<R | null>(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const mounted = useRef(true);
  const notified = useRef('');
  const submittedRun = useRef('');
  useEffect(() => {mounted.current = true; return () => {mounted.current = false;};}, []);
  useEffect(() => {
    if (!initialRunId) return;
    let active = true;
    api.view(initialRunId).then((value) => {
      if (!active) return;
      if (value.source_fit_id !== fit || value.run_id !== initialRunId) {setError('This run belongs to another source version.'); return;}
      if (value.status === 'succeeded' && submittedRun.current !== value.run_id) notified.current = value.run_id;
      setRun(value);
    }).catch((reason) => {if (active) setError(errorText(reason));});
    return () => {active = false;};
  }, [initialRunId, fit, api]);
  useEffect(() => {
    if (run?.status === 'succeeded' && notified.current !== run.run_id) {
      notified.current = run.run_id;
      onCompleted?.();
    }
  }, [run, onCompleted]);
  const activeRunId = run?.run_id;
  const activeStatus = run?.status;
  const controlAvailable = run?.control_available !== false;
  useEffect(() => {
    if (!activeRunId || !activeStatus || !controlAvailable || !['pending', 'running'].includes(activeStatus)) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const value = await api.view(activeRunId!);
        if (!active || value.source_fit_id !== fit || value.run_id !== activeRunId) return;
        setRun(value);
        setError('');
        if (['pending', 'running'].includes(value.status)) timer = setTimeout(poll, 500);
      } catch (reason) {
        if (active) {setError(errorText(reason)); timer = setTimeout(poll, 1000);}
      }
    }
    timer = setTimeout(poll, 100);
    return () => {active = false; clearTimeout(timer);};
  }, [activeRunId, activeStatus, controlAvailable, fit, api]);
  async function start(payload: P) {
    setSubmitting(true); setError('');
    try {
      const result = await api.submit(fit, payload);
      if (mounted.current && result.source_fit_id === fit) {submittedRun.current = result.run_id; setRun(result); onRun?.(result.run_id);}
    } catch (reason) {if (mounted.current) setError(errorText(reason));}
    finally {if (mounted.current) setSubmitting(false);}
  }
  async function cancel() {
    if (!run) return;
    try {
      const result = await api.cancel(run.run_id);
      if (mounted.current && result.source_fit_id === fit) setRun(result);
    } catch (reason) {if (mounted.current) setError(errorText(reason));}
  }
  return {run, error, submitting, controlAvailable, start, cancel};
}
