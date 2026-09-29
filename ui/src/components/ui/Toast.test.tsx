/**
 * Toast stacking/dedup/a11y tests (issue #7428).
 */

import { render, screen, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { ToastProvider, useToast, type ToastType } from './Toast';

// readCachedSettings is consulted for notification prefs; default to "all".
vi.mock('@/api/settingsClient', () => ({
  readCachedSettings: () => null,
}));

function Emitter({
  onReady,
}: {
  onReady: (api: ReturnType<typeof useToast>) => void;
}) {
  const api = useToast();
  onReady(api);
  return null;
}

function setup() {
  let api: ReturnType<typeof useToast> | null = null;
  render(
    <ToastProvider>
      <Emitter onReady={(a) => (api = a)} />
    </ToastProvider>,
  );
  return () => api!;
}

function emit(getApi: () => ReturnType<typeof useToast>, msg: string, type: ToastType) {
  act(() => {
    getApi().showToast(msg, type);
  });
}

describe('ToastProvider', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('caps the visible stack at 5, dropping the oldest', () => {
    const getApi = setup();
    for (let i = 0; i < 10; i++) {
      emit(getApi, `error ${i}`, 'error');
    }
    const alerts = screen.getAllByRole('alert');
    expect(alerts.length).toBe(5);
    // Oldest (error 0..4) dropped; newest retained.
    expect(screen.queryByText('error 0')).not.toBeInTheDocument();
    expect(screen.getByText('error 9')).toBeInTheDocument();
  });

  it('coalesces duplicate (message, type) toasts with a count badge', () => {
    const getApi = setup();
    emit(getApi, 'backend down', 'error');
    emit(getApi, 'backend down', 'error');
    emit(getApi, 'backend down', 'error');

    // One toast, not three.
    expect(screen.getAllByText('backend down')).toHaveLength(1);
    expect(screen.getByText('×3')).toBeInTheDocument();
  });

  it('announces error toasts assertively and others politely', () => {
    const getApi = setup();
    emit(getApi, 'boom', 'error');
    emit(getApi, 'fyi', 'info');

    const errorToast = screen.getByText('boom').closest('[aria-live]');
    expect(errorToast).toHaveAttribute('role', 'alert');
    expect(errorToast).toHaveAttribute('aria-live', 'assertive');

    const infoToast = screen.getByText('fyi').closest('[aria-live]');
    expect(infoToast).toHaveAttribute('role', 'status');
    expect(infoToast).toHaveAttribute('aria-live', 'polite');
  });
});
