/**
 * Tests for SimulationToolbar component.
 *
 * See issue #1179
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';

import { SimulationToolbar } from './SimulationToolbar';

describe('SimulationToolbar', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders tool mode buttons', () => {
      render(<SimulationToolbar isRunning={false} />);

      expect(screen.getByLabelText('Select mode')).toBeInTheDocument();
      expect(screen.getByLabelText('Position mode')).toBeInTheDocument();
      expect(screen.getByLabelText('Rotate mode')).toBeInTheDocument();
      expect(screen.getByLabelText('Measure mode')).toBeInTheDocument();
    });

    it('renders toggle buttons', () => {
      render(<SimulationToolbar isRunning={false} />);

      expect(
        screen.getByLabelText('Show force overlays'),
      ).toBeInTheDocument();
      expect(
        screen.getByLabelText('Show joint angles'),
      ).toBeInTheDocument();
    });

    it('renders toolbar with proper ARIA role', () => {
      render(<SimulationToolbar isRunning={false} />);

      expect(
        screen.getByRole('toolbar', { name: 'Simulation tools' }),
      ).toBeInTheDocument();
    });
  });

  describe('tool mode switching', () => {
    it('calls onToolModeChange when mode changes', () => {
      const onToolModeChange = vi.fn();
      render(
        <SimulationToolbar
          isRunning={false}
          onToolModeChange={onToolModeChange}
        />,
      );

      fireEvent.click(screen.getByLabelText('Position mode'));
      expect(onToolModeChange).toHaveBeenCalledWith('position');

      fireEvent.click(screen.getByLabelText('Measure mode'));
      expect(onToolModeChange).toHaveBeenCalledWith('measure');
    });

    it('highlights active mode button', () => {
      render(<SimulationToolbar isRunning={false} />);

      const selectButton = screen.getByLabelText('Select mode');
      expect(selectButton).toHaveAttribute('aria-pressed', 'true');

      fireEvent.click(screen.getByLabelText('Position mode'));
      expect(screen.getByLabelText('Position mode')).toHaveAttribute(
        'aria-pressed',
        'true',
      );
      expect(screen.getByLabelText('Select mode')).toHaveAttribute(
        'aria-pressed',
        'false',
      );
    });
  });

  describe('force overlay toggle', () => {
    it('calls onForceOverlayToggle when toggled', () => {
      const onForceOverlayToggle = vi.fn();
      render(
        <SimulationToolbar
          isRunning={false}
          onForceOverlayToggle={onForceOverlayToggle}
        />,
      );

      fireEvent.click(screen.getByLabelText('Show force overlays'));
      expect(onForceOverlayToggle).toHaveBeenCalledWith(true);
    });

    it('updates aria-pressed state on toggle', () => {
      render(<SimulationToolbar isRunning={false} />);

      const forcesButton = screen.getByLabelText('Show force overlays');
      expect(forcesButton).toHaveAttribute('aria-pressed', 'false');

      fireEvent.click(forcesButton);
      // After click, button should now say "Hide force overlays"
      expect(
        screen.getByLabelText('Hide force overlays'),
      ).toHaveAttribute('aria-pressed', 'true');
    });
  });

  describe('joint angles display', () => {
    it('does not show joint angles by default', () => {
      render(<SimulationToolbar isRunning={false} />);

      expect(screen.queryByText('Joint Angles')).not.toBeInTheDocument();
    });

    it('toggles joint angles display', () => {
      render(<SimulationToolbar isRunning={false} />);

      fireEvent.click(screen.getByLabelText('Show joint angles'));
      // Even though there is no data yet, the toggle state changes
      expect(
        screen.getByLabelText('Hide joint angles'),
      ).toBeInTheDocument();
    });
  });

  describe('SimulationToolbar polling (#8941)', () => {
    let fetchMock: ReturnType<typeof vi.fn>;

    function measurementsResponse() {
      return {
        ok: true,
        status: 200,
        headers: new Headers(),
        json: () =>
          Promise.resolve({
            joint_angles: [
              {
                joint_name: 'shoulder',
                angle_rad: 0.5,
                angle_deg: 28.6,
                velocity: 0.1,
                torque: 1.2,
              },
            ],
            measurements: [],
          }),
      };
    }

    async function flush() {
      await act(async () => {
        for (let i = 0; i < 5; i++) await Promise.resolve();
      });
    }

    beforeEach(() => {
      vi.useFakeTimers();
      fetchMock = vi.fn().mockResolvedValue(measurementsResponse());
      global.fetch = fetchMock as unknown as typeof fetch;
    });

    afterEach(() => {
      vi.useRealTimers();
      Object.defineProperty(document, 'visibilityState', {
        configurable: true,
        get: () => 'visible',
      });
    });

    it('does not poll while showJoints is false', async () => {
      render(<SimulationToolbar isRunning={true} />);
      await act(async () => {
        vi.advanceTimersByTime(5000);
      });
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('does not poll while simulation is stopped', async () => {
      render(<SimulationToolbar isRunning={false} />);
      fireEvent.click(screen.getByLabelText('Show joint angles'));
      await act(async () => {
        vi.advanceTimersByTime(5000);
      });
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('polls periodically when showJoints and isRunning are both true', async () => {
      render(<SimulationToolbar isRunning={true} pollInterval={1000} />);
      fireEvent.click(screen.getByLabelText('Show joint angles'));
      await flush();
      expect(fetchMock).toHaveBeenCalledTimes(1);

      for (let tick = 1; tick <= 3; tick++) {
        await act(async () => {
          vi.advanceTimersByTime(1000);
        });
        await flush();
      }
      // 1 immediate + 3 ticks = 4 calls over 3 seconds
      expect(fetchMock).toHaveBeenCalledTimes(4);
      expect(String(fetchMock.mock.calls[0][0])).toContain('/api/simulation/measurements');
    });

    it('pauses polling while the tab is hidden', async () => {
      render(<SimulationToolbar isRunning={true} pollInterval={1000} />);
      fireEvent.click(screen.getByLabelText('Show joint angles'));
      await flush();
      expect(fetchMock).toHaveBeenCalledTimes(1);

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
      // Stays paused at 1 call while hidden
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    it('clears interval on unmount', async () => {
      const { unmount } = render(<SimulationToolbar isRunning={true} pollInterval={1000} />);
      fireEvent.click(screen.getByLabelText('Show joint angles'));
      await flush();
      expect(fetchMock).toHaveBeenCalledTimes(1);

      unmount();
      await act(async () => {
        vi.advanceTimersByTime(5000);
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });
  });
});
