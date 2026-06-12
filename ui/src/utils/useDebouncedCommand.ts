/**
 * useDebouncedCommand — debounce a slider/range value into a single network
 * command, with out-of-order protection and failure reverting (issue #7425).
 *
 * UI state should update immediately for responsiveness; the network call is
 * debounced so a drag does not fire one request per tick. A monotonically
 * increasing request id guards against out-of-order responses applying a stale
 * value, and a failed send reverts the displayed value to the last confirmed
 * backend value.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

export interface CommandResult {
  success: boolean;
  error?: string;
}

export interface DebouncedCommand {
  /** Current displayed value (optimistic while a send is pending). */
  value: number;
  /** Update the displayed value and schedule a debounced send. */
  setValue: (next: number) => void;
  /** True while a debounce/send is in flight. */
  pending: boolean;
}

/**
 * @param confirmedValue The last value confirmed by the backend (source of
 *   truth); the displayed value follows it whenever no edit is pending.
 * @param send Performs the network command; must report failure in-band.
 * @param onError Called with a message when a send fails.
 * @param delayMs Debounce window (default 200ms).
 */
export function useDebouncedCommand(
  confirmedValue: number,
  send: (value: number) => Promise<CommandResult>,
  onError: (message: string) => void,
  delayMs = 200,
): DebouncedCommand {
  // Optimistic in-flight value; null means "follow the confirmed value". Using
  // a derived display value (editValue ?? confirmedValue) avoids a setState in
  // an effect to track external refreshes.
  const [editValue, setEditValue] = useState<number | null>(null);

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const requestIdRef = useRef(0);
  const latestAppliedRef = useRef(0);
  // Keep stable refs so the debounced closure never goes stale. Updated in an
  // effect (not during render) to satisfy react-hooks/refs.
  const sendRef = useRef(send);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    sendRef.current = send;
    onErrorRef.current = onError;
  }, [send, onError]);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const setValue = useCallback(
    (next: number) => {
      setEditValue(next);
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      timerRef.current = setTimeout(() => {
        const id = ++requestIdRef.current;
        void sendRef.current(next).then((result) => {
          // Ignore stale responses: only the most recent request may apply.
          if (id < latestAppliedRef.current) {
            return;
          }
          latestAppliedRef.current = id;
          if (!result.success) {
            onErrorRef.current(result.error ?? 'Command failed');
          }
          // On success the confirmed value advances and the optimistic value is
          // dropped; on failure we also drop it, reverting to confirmed.
          if (id === requestIdRef.current) {
            setEditValue(null);
          }
        });
      }, delayMs);
    },
    [delayMs],
  );

  return {
    value: editValue ?? confirmedValue,
    setValue,
    pending: editValue !== null,
  };
}
