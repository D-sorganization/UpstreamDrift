/**
 * Tests for ForceOverlayPanel polling cadence and gating.
 *
 * See issues #1199, #8941
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { ForceOverlayPanel } from './ForceOverlayPanel';

function forcesResponse() {
  return {
    ok: true,
    status: 200,
    headers: new Headers(),
    json: () =>
      Promise.resolve({ vectors: [], total_force_magnitude: 1, total_torque_magnitude: 2 }),
  };
}

async function flush() {
  await act(async () => {
    for (let i = 0; i < 5; i++) await Promise.resolve();
  });
}

function enableOverlay() {
  fireEvent.click(screen.getByRole('checkbox', { name: /off/i }));
}

describe('ForceOverlayPanel polling (#8941)', () => {
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    fetchMock = vi.fn().mockResolvedValue(forcesResponse());
    global.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    vi.useRealTimers();
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'visible',
    });
  });

  it('does not poll while the overlay is off', async () => {
    render(<ForceOverlayPanel onVectorsChange={vi.fn()} isRunning={true} />);
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('does not poll while the simulation is stopped', async () => {
    render(<ForceOverlayPanel onVectorsChange={vi.fn()} isRunning={false} />);
    enableOverlay();
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('polls at 2 Hz by default (was 5 Hz)', async () => {
    render(<ForceOverlayPanel onVectorsChange={vi.fn()} isRunning={true} />);
    enableOverlay();
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(1);

    for (let tick = 1; tick <= 4; tick++) {
      await act(async () => {
        vi.advanceTimersByTime(500);
      });
      await flush();
    }
    // 1 immediate + 4 ticks over 2 s; the old 200 ms loop would have made 10.
    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(String(fetchMock.mock.calls[0][0])).toContain('/api/simulation/forces?');
  });

  it('pauses while the tab is hidden', async () => {
    render(<ForceOverlayPanel onVectorsChange={vi.fn()} isRunning={true} />);
    enableOverlay();
    await flush();
    act(() => {
      Object.defineProperty(document, 'visibilityState', {
        configurable: true,
        get: () => 'hidden',
      });
      document.dispatchEvent(new Event('visibilitychange'));
    });
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('clears its interval on unmount', async () => {
    const { unmount } = render(
      <ForceOverlayPanel onVectorsChange={vi.fn()} isRunning={true} />,
    );
    enableOverlay();
    await flush();
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('uses socket payload when connected and disables REST polling', async () => {
    const onGlyphsChange = vi.fn();
    const fakeGlyphs = {
      schema_version: 'glyph-set-v1',
      time_s: 0.1,
      arrows: [],
      torque_arcs: [],
      legend: {
        force_reference_n: null,
        force_reference_length_m: null,
        torque_reference_nm: null,
        torque_reference_radius_m: null,
        kinds_present: [],
        unavailable_labels: [],
        engine: 'test',
        source_labels: [],
      },
    };

    render(
      <ForceOverlayPanel
        onGlyphsChange={onGlyphsChange}
        isRunning={true}
        isConnected={true}
        socketForceOverlay={fakeGlyphs}
      />,
    );
    enableOverlay();
    await flush();

    // Verify socket payload was received
    expect(onGlyphsChange).toHaveBeenCalledWith(fakeGlyphs);

    // Advance time - REST fetch should NOT be called because socket is connected
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('falls back to REST polling while socket is disconnected', async () => {
    const onGlyphsChange = vi.fn();
    render(
      <ForceOverlayPanel
        onGlyphsChange={onGlyphsChange}
        isRunning={true}
        isConnected={false}
      />,
    );
    enableOverlay();
    await flush();

    expect(fetchMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(500);
    });
    await flush();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
