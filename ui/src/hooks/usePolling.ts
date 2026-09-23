/**
 * usePolling — the single REST polling loop shared by the Simulation-page
 * panels (issue #8941).
 *
 * Guarantees:
 *  - at most one request per tick: a tick is skipped while the previous task
 *    is still pending, so slow responses never pile up;
 *  - no polling while `enabled` is false (e.g. the simulation is stopped or the
 *    overlay is switched off) or while the browser tab is hidden;
 *  - the interval is cleared on disable, on hide and on unmount (no leaks);
 *  - the latest `task` is always called without restarting the interval.
 */

import { useEffect, useRef, useSyncExternalStore } from 'react';

export interface UsePollingOptions {
  /** Tick period in ms. Contract: a finite number > 0. */
  intervalMs: number;
  /** Poll only while true (e.g. simulation running and panel active). */
  enabled: boolean;
  /** Pause while `document.visibilityState === 'hidden'` (default true). */
  pauseWhenHidden?: boolean;
  /** Run the task as soon as polling starts or resumes (default true). */
  immediate?: boolean;
}

/**
 * Precondition check shared by the polling hooks.
 *
 * @throws RangeError when `intervalMs` is not a finite number > 0.
 */
export function assertValidInterval(intervalMs: number): void {
  if (!Number.isFinite(intervalMs) || intervalMs <= 0) {
    throw new RangeError(
      `usePolling: intervalMs must be a finite number > 0 (got ${intervalMs})`,
    );
  }
}

function subscribeVisibility(onChange: () => void): () => void {
  document.addEventListener('visibilitychange', onChange);
  return () => document.removeEventListener('visibilitychange', onChange);
}

function isDocumentVisible(): boolean {
  return document.visibilityState !== 'hidden';
}

function alwaysVisible(): boolean {
  return true;
}

/** Whether the page is currently visible (re-renders on change). */
function useDocumentVisible(): boolean {
  return useSyncExternalStore(subscribeVisibility, isDocumentVisible, alwaysVisible);
}

/**
 * Run `task` every `intervalMs` while polling is active.
 *
 * @param task - Work for one tick; a returned promise marks the tick in flight.
 * @param options - See {@link UsePollingOptions}.
 */
export function usePolling(
  task: () => Promise<unknown> | unknown,
  { intervalMs, enabled, pauseWhenHidden = true, immediate = true }: UsePollingOptions,
): void {
  assertValidInterval(intervalMs);

  const taskRef = useRef(task);
  useEffect(() => {
    taskRef.current = task;
  }, [task]);

  const visible = useDocumentVisible();
  const active = enabled && (visible || !pauseWhenHidden);

  useEffect(() => {
    if (!active) {
      return undefined;
    }
    let inFlight = false;
    const tick = () => {
      if (inFlight) {
        return;
      }
      inFlight = true;
      let pending: Promise<unknown>;
      try {
        pending = Promise.resolve(taskRef.current());
      } catch (err) {
        pending = Promise.reject(err);
      }
      pending
        .catch(() => {
          // The task owns its error reporting; a rejection must not stop polling.
        })
        .finally(() => {
          inFlight = false;
        });
    };
    if (immediate) {
      tick();
    }
    const id = setInterval(tick, intervalMs);
    return () => clearInterval(id);
  }, [active, intervalMs, immediate]);
}
