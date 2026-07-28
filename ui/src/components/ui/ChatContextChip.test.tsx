/**
 * Tests for ChatContextChip (issue #7453).
 *
 * Mocks fetch to drive the GET /api/chat/context payload and asserts the
 * chip's render states: context present, context absent, fetch failure.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, waitFor } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { ChatContextChip, formatContextLabel } from './ChatContextChip';
import type { ChatContextInfo } from './ChatContextChip';

function mockFetchWith(payload: unknown, ok = true) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    json: async () => payload,
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

const FULL_CONTEXT: ChatContextInfo = {
  engines_loaded: ['mujoco', 'pendulum'],
  active_engine: 'mujoco',
  active_model: 'golf_swing.urdf',
  simulation: {
    engine: 'mujoco',
    model: 'golf_swing.urdf',
    duration_seconds: 3.0,
    status: 'completed',
  },
};

const EMPTY_CONTEXT: ChatContextInfo = {
  engines_loaded: [],
  active_engine: null,
  active_model: null,
  simulation: null,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('ChatContextChip', () => {
  it('renders engine, model, and last-run duration when context is present', async () => {
    const fetchMock = mockFetchWith(FULL_CONTEXT);
    render(<ChatContextChip refreshMs={0} />);

    const chip = await screen.findByTestId('chat-context-chip');
    expect(chip.textContent).toContain('mujoco');
    expect(chip.textContent).toContain('golf_swing.urdf');
    expect(chip.textContent).toContain('last run 3.0s');
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('/api/chat/context'),
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('renders nothing when the context payload is empty', async () => {
    const fetchMock = mockFetchWith(EMPTY_CONTEXT);
    render(<ChatContextChip refreshMs={0} />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.queryByTestId('chat-context-chip')).toBeNull();
  });

  it('renders nothing when the fetch fails', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error('network down'));
    vi.stubGlobal('fetch', fetchMock);
    render(<ChatContextChip refreshMs={0} />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.queryByTestId('chat-context-chip')).toBeNull();
  });

  it('renders nothing on a non-OK response', async () => {
    const fetchMock = mockFetchWith({}, false);
    render(<ChatContextChip refreshMs={0} />);

    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(screen.queryByTestId('chat-context-chip')).toBeNull();
  });

  it('shows "running" while a simulation is in flight', async () => {
    mockFetchWith({
      ...FULL_CONTEXT,
      simulation: { ...FULL_CONTEXT.simulation!, status: 'running' },
    });
    render(<ChatContextChip refreshMs={0} />);

    const chip = await screen.findByTestId('chat-context-chip');
    expect(chip.textContent).toContain('running 3.0s');
  });
});

describe('formatContextLabel', () => {
  it('returns null for null or empty context', () => {
    expect(formatContextLabel(null)).toBeNull();
    expect(formatContextLabel(EMPTY_CONTEXT)).toBeNull();
  });

  it('joins parts with a middle dot', () => {
    expect(formatContextLabel(FULL_CONTEXT)).toBe(
      'mujoco · golf_swing.urdf · last run 3.0s',
    );
  });

  it('falls back to status when duration is missing', () => {
    expect(
      formatContextLabel({
        ...EMPTY_CONTEXT,
        active_engine: 'drake',
        simulation: {
          engine: 'drake',
          model: null,
          duration_seconds: null,
          status: 'failed',
        },
      }),
    ).toBe('drake · failed');
  });
});
