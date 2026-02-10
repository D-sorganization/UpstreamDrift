import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen, fireEvent } from '@testing-library/dom';
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

  // ──────────────────────────────────────────────────────────────
  //  Phase 2: Speed Control Tests (See issue #1202)
  // ──────────────────────────────────────────────────────────────
  describe('speed control', () => {
    it('renders speed slider when onSpeedChange is provided', () => {
      const onSpeedChange = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onSpeedChange={onSpeedChange}
        />
      );

      expect(screen.getByLabelText(/simulation speed/i)).toBeInTheDocument();
    });

    it('does not render speed slider when onSpeedChange is not provided', () => {
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
        />
      );

      expect(screen.queryByLabelText(/simulation speed/i)).not.toBeInTheDocument();
    });

    it('displays speed value', () => {
      const onSpeedChange = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onSpeedChange={onSpeedChange}
          initialSpeed={2.0}
        />
      );

      expect(screen.getByText('2.0x')).toBeInTheDocument();
    });

    it('renders speed preset buttons', () => {
      const onSpeedChange = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onSpeedChange={onSpeedChange}
        />
      );

      expect(screen.getByRole('button', { name: /set speed to 1x/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /set speed to 2x/i })).toBeInTheDocument();
    });

    it('calls onSpeedChange when preset button is clicked', () => {
      const onSpeedChange = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onSpeedChange={onSpeedChange}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /set speed to 2x/i }));
      expect(onSpeedChange).toHaveBeenCalledWith(2.0);
    });
  });

  // ──────────────────────────────────────────────────────────────
  //  Phase 2: Single-Step Tests (See issue #1202)
  // ──────────────────────────────────────────────────────────────
  describe('single step', () => {
    it('renders step button when onStep is provided and running', () => {
      const onStep = vi.fn();
      render(
        <SimulationControls
          isRunning={true}
          isPaused={true}
          {...mockHandlers}
          onStep={onStep}
        />
      );

      expect(screen.getByRole('button', { name: /single step/i })).toBeInTheDocument();
    });

    it('step button is disabled when not paused', () => {
      const onStep = vi.fn();
      render(
        <SimulationControls
          isRunning={true}
          isPaused={false}
          {...mockHandlers}
          onStep={onStep}
        />
      );

      const stepBtn = screen.getByRole('button', { name: /single step/i });
      expect(stepBtn).toBeDisabled();
    });

    it('calls onStep when step button is clicked while paused', () => {
      const onStep = vi.fn();
      render(
        <SimulationControls
          isRunning={true}
          isPaused={true}
          {...mockHandlers}
          onStep={onStep}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /single step/i }));
      expect(onStep).toHaveBeenCalledTimes(1);
    });
  });

  // ──────────────────────────────────────────────────────────────
  //  Phase 2: Camera Preset Tests (See issue #1202)
  // ──────────────────────────────────────────────────────────────
  describe('camera presets', () => {
    it('renders camera preset buttons when onCameraChange is provided', () => {
      const onCameraChange = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onCameraChange={onCameraChange}
        />
      );

      expect(screen.getByRole('button', { name: /side camera view/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /front camera view/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /top camera view/i })).toBeInTheDocument();
    });

    it('calls onCameraChange when preset is clicked', () => {
      const onCameraChange = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onCameraChange={onCameraChange}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /front camera view/i }));
      expect(onCameraChange).toHaveBeenCalledWith('front');
    });
  });

  // ──────────────────────────────────────────────────────────────
  //  Phase 2: Recording Controls Tests (See issue #1202)
  // ──────────────────────────────────────────────────────────────
  describe('recording controls', () => {
    it('renders record button when onRecordingToggle is provided', () => {
      const onRecordingToggle = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onRecordingToggle={onRecordingToggle}
        />
      );

      expect(screen.getByRole('button', { name: /start recording/i })).toBeInTheDocument();
    });

    it('toggles recording state when clicked', () => {
      const onRecordingToggle = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onRecordingToggle={onRecordingToggle}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /start recording/i }));
      expect(onRecordingToggle).toHaveBeenCalledWith(true);
    });

    it('renders export button when onExportTrajectory is provided', () => {
      const onRecordingToggle = vi.fn();
      const onExportTrajectory = vi.fn();
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
          onRecordingToggle={onRecordingToggle}
          onExportTrajectory={onExportTrajectory}
        />
      );

      expect(screen.getByRole('button', { name: /export trajectory/i })).toBeInTheDocument();
    });
  });

  // ──────────────────────────────────────────────────────────────
  //  Phase 2: Stats Display Tests (See issue #1202)
  // ──────────────────────────────────────────────────────────────
  describe('stats display', () => {
    const mockStats = {
      simTime: 2.345,
      fps: 500,
      realTimeFactor: 0.96,
      frameCount: 1172,
    };

    it('renders stats when provided', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={false}
          {...mockHandlers}
          stats={mockStats}
        />
      );

      expect(screen.getByText('2.345s')).toBeInTheDocument();
      expect(screen.getByText('500')).toBeInTheDocument();
      expect(screen.getByText('0.96x')).toBeInTheDocument();
      expect(screen.getByText('1172')).toBeInTheDocument();
    });

    it('has accessible status role', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={false}
          {...mockHandlers}
          stats={mockStats}
        />
      );

      expect(screen.getByRole('status', { name: /simulation statistics/i })).toBeInTheDocument();
    });

    it('does not render stats when not provided', () => {
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
        />
      );

      expect(screen.queryByRole('status')).not.toBeInTheDocument();
    });
  });

  // ──────────────────────────────────────────────────────────────
  //  Phase 2: Keyboard Shortcuts Tests (See issue #1202)
  // ──────────────────────────────────────────────────────────────
  describe('keyboard shortcuts', () => {
    it('Space starts simulation when not running', () => {
      render(
        <SimulationControls
          isRunning={false}
          {...mockHandlers}
        />
      );

      fireEvent.keyDown(window, { code: 'Space' });
      expect(mockHandlers.onStart).toHaveBeenCalledTimes(1);
    });

    it('Space resumes when paused', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={true}
          {...mockHandlers}
        />
      );

      fireEvent.keyDown(window, { code: 'Space' });
      expect(mockHandlers.onResume).toHaveBeenCalledTimes(1);
    });

    it('Escape stops simulation', () => {
      render(
        <SimulationControls
          isRunning={true}
          isPaused={false}
          {...mockHandlers}
        />
      );

      fireEvent.keyDown(window, { code: 'Escape' });
      expect(mockHandlers.onStop).toHaveBeenCalledTimes(1);
    });
  });
});
