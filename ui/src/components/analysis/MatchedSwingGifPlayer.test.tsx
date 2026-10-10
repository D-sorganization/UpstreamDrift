/**
 * Tests for MatchedSwingGifPlayer — frame-exact GIF playback (#11987).
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen } from '@testing-library/react';
import { MatchedSwingGifPlayer } from './MatchedSwingGifPlayer';
import * as api from '@/api/matchedSwings';
import type { MatchedSwingAnimationInfo } from '@/api/matchedSwings';

const INFO: MatchedSwingAnimationInfo = {
  schema_version: 'matched-swing-animation/1',
  frame_count: 3,
  durations_ms: [40, 100, 120],
  width: 4,
  height: 3,
};

/** Flush resolved promises queued by the pending fetch (no timer advance). */
async function flush() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

describe('MatchedSwingGifPlayer', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it('shows the no-animation message and disables every control', () => {
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation={false} />);
    expect(
      screen.getByText('No animation artefact for this run.'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pause' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Restart' })).toBeDisabled();
  });

  it('does not fetch animation info when there is no animation artefact', () => {
    const spy = vi.spyOn(api, 'fetchMatchedSwingAnimationInfo');
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation={false} />);
    expect(spy).not.toHaveBeenCalled();
  });

  it('shows an error and disables controls when the info fetch fails', async () => {
    vi.spyOn(api, 'fetchMatchedSwingAnimationInfo').mockRejectedValue(
      new Error('boom'),
    );
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation />);
    await flush();

    expect(screen.getByText('Animation unavailable: boom')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
  });

  it('enables controls and shows the frame counter once info loads', async () => {
    vi.spyOn(api, 'fetchMatchedSwingAnimationInfo').mockResolvedValue(INFO);
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation />);
    await flush();

    expect(screen.getByText('Frame 1 / 3')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Play' })).toBeEnabled();
  });

  it('advances frames on a timer while playing', async () => {
    vi.spyOn(api, 'fetchMatchedSwingAnimationInfo').mockResolvedValue(INFO);
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation />);
    await flush();
    expect(screen.getByText('Frame 1 / 3')).toBeInTheDocument();

    // Frame 0's duration is 40ms.
    await act(async () => {
      vi.advanceTimersByTime(40);
    });
    expect(screen.getByText('Frame 2 / 3')).toBeInTheDocument();

    // Frame 1's duration is 100ms.
    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText('Frame 3 / 3')).toBeInTheDocument();

    // Frame 2's duration is 120ms; looping wraps back to frame 1.
    await act(async () => {
      vi.advanceTimersByTime(120);
    });
    expect(screen.getByText('Frame 1 / 3')).toBeInTheDocument();
  });

  it('Pause freezes the current frame', async () => {
    vi.spyOn(api, 'fetchMatchedSwingAnimationInfo').mockResolvedValue(INFO);
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation />);
    await flush();

    await act(async () => {
      vi.advanceTimersByTime(40);
    });
    expect(screen.getByText('Frame 2 / 3')).toBeInTheDocument();

    act(() => {
      screen.getByRole('button', { name: 'Pause' }).click();
    });
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(screen.getByText('Frame 2 / 3')).toBeInTheDocument();
  });

  it('Play resumes from the paused frame', async () => {
    vi.spyOn(api, 'fetchMatchedSwingAnimationInfo').mockResolvedValue(INFO);
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation />);
    await flush();

    await act(async () => {
      vi.advanceTimersByTime(40);
    });
    expect(screen.getByText('Frame 2 / 3')).toBeInTheDocument();

    act(() => {
      screen.getByRole('button', { name: 'Pause' }).click();
    });
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(screen.getByText('Frame 2 / 3')).toBeInTheDocument();

    act(() => {
      screen.getByRole('button', { name: 'Play' }).click();
    });
    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText('Frame 3 / 3')).toBeInTheDocument();
  });

  it('Restart returns to frame 1 of N and plays', async () => {
    vi.spyOn(api, 'fetchMatchedSwingAnimationInfo').mockResolvedValue(INFO);
    render(<MatchedSwingGifPlayer runId="r1" hasAnimation />);
    await flush();

    await act(async () => {
      vi.advanceTimersByTime(40);
    });
    await act(async () => {
      vi.advanceTimersByTime(100);
    });
    expect(screen.getByText('Frame 3 / 3')).toBeInTheDocument();

    act(() => {
      screen.getByRole('button', { name: 'Pause' }).click();
      screen.getByRole('button', { name: 'Restart' }).click();
    });
    expect(screen.getByText('Frame 1 / 3')).toBeInTheDocument();

    // Restart also resumes playback.
    await act(async () => {
      vi.advanceTimersByTime(40);
    });
    expect(screen.getByText('Frame 2 / 3')).toBeInTheDocument();
  });
});
