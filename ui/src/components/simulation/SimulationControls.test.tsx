import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SimulationControls } from './SimulationControls';

describe('SimulationControls', () => {
  const mockHandlers = {
    onStart: vi.fn(),
    onStop: vi.fn(),
    onPause: vi.fn(),
    onResume: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('when simulation is not running', () => {
    it('renders start button', () => {
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
        />
      );

      expect(screen.getByRole('button', { name: /start simulation/i })).toBeInTheDocument();
    });

    it('calls onStart when start button is clicked', () => {
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));
      expect(mockHandlers.onStart).toHaveBeenCalledTimes(1);
    });
  });

  describe('when simulation is running', () => {
    it('renders pause and stop buttons', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={false}
          {...mockHandlers}
        />
      );

      expect(screen.getByRole('button', { name: /pause simulation/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /stop simulation/i })).toBeInTheDocument();
    });

    it('calls onPause when pause button is clicked', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={false}
          {...mockHandlers}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /pause simulation/i }));
      expect(mockHandlers.onPause).toHaveBeenCalledTimes(1);
    });

    it('calls onStop when stop button is clicked', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={false}
          {...mockHandlers}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /stop simulation/i }));
      expect(mockHandlers.onStop).toHaveBeenCalledTimes(1);
    });
  });

  describe('when simulation is paused', () => {
    it('renders resume button instead of pause', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={true}
          {...mockHandlers}
        />
      );

      expect(screen.getByRole('button', { name: /resume simulation/i })).toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /pause simulation/i })).not.toBeInTheDocument();
    });

    it('calls onResume when resume button is clicked', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={true}
          {...mockHandlers}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /resume simulation/i }));
      expect(mockHandlers.onResume).toHaveBeenCalledTimes(1);
    });
  });

  describe('accessibility', () => {
    it('has toolbar role with accessible name', () => {
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
        />
      );

      expect(screen.getByRole('toolbar', { name: /simulation controls/i })).toBeInTheDocument();
    });

    it('buttons have visible focus rings', () => {
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
        />
      );

      const button = screen.getByRole('button', { name: /start simulation/i });
      expect(button.className).toContain('focus:ring');
    });
  });
});
