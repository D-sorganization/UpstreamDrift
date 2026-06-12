/**
 * useDebouncedCommand tests (issue #7425).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useDebouncedCommand } from './useDebouncedCommand';

describe('useDebouncedCommand', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('coalesces rapid changes into a single send with the final value', async () => {
    const send = vi.fn().mockResolvedValue({ success: true });
    const { result } = renderHook(() =>
      useDebouncedCommand(1, send, vi.fn(), 200),
    );

    act(() => {
      result.current.setValue(2);
      result.current.setValue(3);
      result.current.setValue(4);
    });
    // Display updates immediately; no send yet.
    expect(result.current.value).toBe(4);
    expect(send).not.toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(200);
    });
    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith(4);
  });

  it('reverts the displayed value and reports on send failure', async () => {
    const onError = vi.fn();
    const send = vi
      .fn()
      .mockResolvedValue({ success: false, error: 'boom' });
    const { result } = renderHook(() =>
      useDebouncedCommand(1, send, onError, 200),
    );

    act(() => {
      result.current.setValue(9);
    });
    await act(async () => {
      vi.advanceTimersByTime(200);
    });

    expect(onError).toHaveBeenCalledWith('boom');
    // Reverted to the last confirmed backend value.
    expect(result.current.value).toBe(1);
  });

  it('does not send when unmounted during the debounce window', async () => {
    const send = vi.fn().mockResolvedValue({ success: true });
    const { result, unmount } = renderHook(() =>
      useDebouncedCommand(1, send, vi.fn(), 200),
    );

    act(() => {
      result.current.setValue(5);
    });
    unmount();
    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    expect(send).not.toHaveBeenCalled();
  });

  it('follows the confirmed value when nothing is pending', () => {
    const send = vi.fn().mockResolvedValue({ success: true });
    const { result, rerender } = renderHook(
      ({ confirmed }) => useDebouncedCommand(confirmed, send, vi.fn(), 200),
      { initialProps: { confirmed: 1 } },
    );
    expect(result.current.value).toBe(1);

    rerender({ confirmed: 7 });
    expect(result.current.value).toBe(7);
  });
});
