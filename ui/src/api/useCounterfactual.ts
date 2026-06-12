/**
 * Counterfactual analysis hook (issue #7450).
 *
 * Drives the async-task pattern shared with /simulate/async:
 *  1. GET  /api/analysis/counterfactual/kinds   — engine capability gating
 *  2. POST /api/analysis/counterfactual         — start a task
 *  3. GET  /api/simulate/status/{task_id}       — poll until completed/failed
 *
 * The result payload is the serialized `CounterfactualResult` produced by
 * the Python `AnalysisOrchestrator` (same compute path as the PyQt6
 * dashboard's post-hoc analysis).
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { apiFetch } from './fetch';

// ── Types (mirror src/shared/python/analysis/plot_data.py) ────────────

export type CounterfactualKind =
  | 'ztcf'
  | 'zvcf'
  | 'gravity'
  | 'drift'
  | 'control'
  | 'total';

export interface CounterfactualResult {
  kind: string;
  times: number[];
  values: number[][];
  units: string;
  metadata: Record<string, unknown>;
}

export interface CounterfactualSupport {
  kinds: string[];
  engine: string | null;
  session_available: boolean;
}

interface TaskStartResponse {
  task_id: string;
  status: string;
  kind: string;
}

interface TaskStatusResponse {
  status: 'started' | 'running' | 'completed' | 'failed';
  result?: CounterfactualResult;
  error?: string;
}

export type CounterfactualRunState =
  | 'idle'
  | 'starting'
  | 'running'
  | 'completed'
  | 'failed';

const POLL_INTERVAL_MS = 750;
const MAX_POLLS = 400; // ~5 minutes at 750ms

export interface UseCounterfactualResult {
  /** Engine capability info (null until fetched). */
  support: CounterfactualSupport | null;
  /** Refresh the capability info. */
  fetchSupport: () => Promise<void>;
  /** Run state of the current/last task. */
  runState: CounterfactualRunState;
  /** Result of the last completed run. */
  result: CounterfactualResult | null;
  /** Error message of the last failure (HTTP or task-level). */
  error: string | null;
  /** Start a counterfactual analysis and poll until it finishes. */
  run: (kind: string) => Promise<void>;
}

export function useCounterfactual(): UseCounterfactualResult {
  const [support, setSupport] = useState<CounterfactualSupport | null>(null);
  const [runState, setRunState] = useState<CounterfactualRunState>('idle');
  const [result, setResult] = useState<CounterfactualResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const fetchSupport = useCallback(async () => {
    try {
      const data = await apiFetch<CounterfactualSupport>(
        '/api/analysis/counterfactual/kinds',
      );
      if (isMountedRef.current) setSupport(data);
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch capabilities');
      }
    }
  }, []);

  const run = useCallback(async (kind: string) => {
    setRunState('starting');
    setError(null);
    setResult(null);
    try {
      const started = await apiFetch<TaskStartResponse>(
        '/api/analysis/counterfactual',
        { method: 'POST', body: JSON.stringify({ kind }) },
      );
      if (!isMountedRef.current) return;
      setRunState('running');

      for (let i = 0; i < MAX_POLLS; i++) {
        const status = await apiFetch<TaskStatusResponse>(
          `/api/simulate/status/${started.task_id}`,
        );
        if (!isMountedRef.current) return;
        if (status.status === 'completed' && status.result) {
          setResult(status.result);
          setRunState('completed');
          return;
        }
        if (status.status === 'failed') {
          setError(status.error ?? 'Counterfactual analysis failed');
          setRunState('failed');
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
        if (!isMountedRef.current) return;
      }
      setError('Timed out waiting for the counterfactual task');
      setRunState('failed');
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Counterfactual run failed');
        setRunState('failed');
      }
    }
  }, []);

  return { support, fetchSupport, runState, result, error, run };
}
