/**
 * Tests for SimulationToolbar component.
 *
 * See issue #1179
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

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
});
