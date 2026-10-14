/**
 * BunkerShot3D designer workbench client (#9545, slice 3).
 *
 * Typed client for the versioned route in `src/api/routes/bunker_workbench.py`:
 * `POST /tools/bunker-workbench/v1/evaluate` (one bounded nominal F0 solve) and
 * `POST /tools/bunker-workbench/v1/compare` (the A/B comparison with the
 * model's carry-objective standing). The route adds no physics; neither does
 * this client. It only refuses to promote an unqualified lead to a ranking.
 */

import { apiFetch } from './fetch';
import type {
  BunkerWorkbenchCompareRequestV1,
  BunkerWorkbenchComparisonV1,
  BunkerWorkbenchEvaluationV1,
  BunkerWorkbenchRequestV1,
} from './generated/types';

export type {
  BunkerDesignV1,
  BunkerWorkbenchCompareRequestV1,
  BunkerWorkbenchComparisonV1,
  BunkerWorkbenchEvaluationV1,
  BunkerWorkbenchRequestV1,
} from './generated/types';

/** Versioned route prefix, mounted under the legacy `/api` prefix. */
export const BUNKER_WORKBENCH_PATH = '/api/tools/bunker-workbench/v1';

/** One nominal shot: seconds, not minutes. */
export const EVALUATE_TIMEOUT_MS = 60_000;

/** Two nominal shots plus the fixed playability sweep per design (52 shots). */
export const COMPARE_TIMEOUT_MS = 300_000;

function post<T>(
  path: string,
  body: unknown,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<T> {
  return apiFetch<T>(`${BUNKER_WORKBENCH_PATH}${path}`, {
    method: 'POST',
    body: JSON.stringify(body),
    timeoutMs,
    signal,
  });
}

/** Run one bounded nominal solve; a 422 refusal rejects with its message. */
export function evaluateBunkerDesign(
  request: BunkerWorkbenchRequestV1,
  signal?: AbortSignal,
): Promise<BunkerWorkbenchEvaluationV1> {
  return post(`/evaluate`, request, EVALUATE_TIMEOUT_MS, signal);
}

/** Compare two differently named soles; a 422 refusal rejects with its message. */
export function compareBunkerDesigns(
  request: BunkerWorkbenchCompareRequestV1,
  signal?: AbortSignal,
): Promise<BunkerWorkbenchComparisonV1> {
  return post(`/compare`, request, COMPARE_TIMEOUT_MS, signal);
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
