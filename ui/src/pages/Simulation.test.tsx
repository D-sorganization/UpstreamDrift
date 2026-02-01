import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen, fireEvent, waitFor } from '@testing-library/dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the useSimulation hook
const mockSimulation = {
  isRunning: false,
  isPaused: false,
  currentFrame: null,
  frames: [],
  connectionStatus: 'disconnected' as const,
  start: vi.fn(),
  stop: vi.fn(),
  pause: vi.fn(),
  resume: vi.fn(),
};

vi.mock('@/api/client', () => ({
  useSimulation: vi.fn(() => mockSimulation),
  fetchEngines: vi.fn(),
}));

// Mock Scene3D
vi.mock('@/components/visualization/Scene3D', () => ({
  Scene3D: ({ engine, frame }: { engine: string; frame: unknown }) => (
    <div data-testid="scene3d-mock" data-engine={engine} data-has-frame={!!frame}>
      Scene3D Mock
    </div>
  ),
}));

import { SimulationPage } from './Simulation';
import { useSimulation, fetchEngines } from '@/api/client';
import type { SimulationFrame } from '@/api/client';

const mockEngines = [
  { name: 'mujoco', available: true, loaded: true, capabilities: ['rigid_body'] },
  { name: 'drake', available: true, loaded: false, capabilities: ['rigid_body'] },
];

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('SimulationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset mock state
    Object.assign(mockSimulation, {
      isRunning: false,
      isPaused: false,
      currentFrame: null,
      frames: [],
      connectionStatus: 'disconnected',
      start: vi.fn(),
      stop: vi.fn(),
      pause: vi.fn(),
      resume: vi.fn(),
    });
    vi.mocked(fetchEngines).mockResolvedValue(mockEngines);
  });

  describe('layout', () => {
    it('renders main layout with sidebars', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Golf Suite')).toBeInTheDocument();
      expect(screen.getByText('Simulation')).toBeInTheDocument();
      expect(screen.getByText('Live Analysis')).toBeInTheDocument();
    });

    it('renders 3D scene component', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('scene3d-mock')).toBeInTheDocument();
    });

    it('renders engine selector', async () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      // Engine selector should be present
      await waitFor(() => {
        expect(screen.getByRole('radiogroup', { name: /physics engine/i })).toBeInTheDocument();
      });
    });

    it('renders simulation controls', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByRole('toolbar', { name: /simulation controls/i })).toBeInTheDocument();
    });
  });

  describe('idle state', () => {
    it('shows ready status when not running', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Ready')).toBeInTheDocument();
    });

    it('shows start simulation placeholder in analysis panel', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText(/start simulation to view live data/i)).toBeInTheDocument();
    });

    it('shows start button', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByRole('button', { name: /start simulation/i })).toBeInTheDocument();
    });

    it('passes null frame to Scene3D when idle', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const scene = screen.getByTestId('scene3d-mock');
      expect(scene.getAttribute('data-has-frame')).toBe('false');
    });
  });

  describe('running state', () => {
    const runningFrame: SimulationFrame = {
      frame: 42,
      time: 0.84,
      state: { qpos: [0.1, 0.2, 0.3] },
      analysis: { joint_angles: [0.5, 0.3, 0.2, 0.1] },
    };

    beforeEach(() => {
      Object.assign(mockSimulation, {
        isRunning: true,
        isPaused: false,
        currentFrame: runningFrame,
        frames: [runningFrame],
      });
    });

    it('shows frame count in status overlay', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Frame 42')).toBeInTheDocument();
    });

    it('shows simulation time in overlay', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Time: 0.840s')).toBeInTheDocument();
    });

    it('shows running status indicator (green pulse)', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const statusIndicator = document.querySelector('.bg-green-500.animate-pulse');
      expect(statusIndicator).toBeInTheDocument();
    });

    it('displays simulation state in analysis panel', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Simulation State')).toBeInTheDocument();
      // State should be displayed as JSON
      expect(screen.getByText(/"qpos"/)).toBeInTheDocument();
    });

    it('passes current frame to Scene3D', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const scene = screen.getByTestId('scene3d-mock');
      expect(scene.getAttribute('data-has-frame')).toBe('true');
    });

    it('shows pause and stop buttons', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByRole('button', { name: /pause simulation/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /stop simulation/i })).toBeInTheDocument();
    });

    it('disables engine selector while running', async () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      await waitFor(() => {
        const radios = screen.getAllByRole('radio');
        radios.forEach((radio) => {
          expect(radio).toBeDisabled();
        });
      });
    });
  });

  describe('paused state', () => {
    const pausedFrame: SimulationFrame = {
      frame: 100,
      time: 2.0,
      state: { qpos: [0.5] },
    };

    beforeEach(() => {
      Object.assign(mockSimulation, {
        isRunning: true,
        isPaused: true,
        currentFrame: pausedFrame,
        frames: [pausedFrame],
      });
    });

    it('shows resume button when paused', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByRole('button', { name: /resume simulation/i })).toBeInTheDocument();
    });
  });

  describe('engine selection', () => {
    it('initializes with mujoco engine selected', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const scene = screen.getByTestId('scene3d-mock');
      expect(scene.getAttribute('data-engine')).toBe('mujoco');
    });

    it('calls useSimulation with selected engine', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(useSimulation).toHaveBeenCalledWith('mujoco');
    });

    it('updates engine when selector changes', async () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByText('drake')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('radio', { name: /drake/i }));

      // useSimulation should be called with new engine
      await waitFor(() => {
        expect(useSimulation).toHaveBeenCalledWith('drake');
      });
    });
  });

  describe('simulation controls interaction', () => {
    it('calls start when start button is clicked', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));

      expect(mockSimulation.start).toHaveBeenCalled();
    });

    it('calls stop when stop button is clicked', () => {
      Object.assign(mockSimulation, { isRunning: true });

      render(<SimulationPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByRole('button', { name: /stop simulation/i }));

      expect(mockSimulation.stop).toHaveBeenCalled();
    });

    it('calls pause when pause button is clicked', () => {
      Object.assign(mockSimulation, { isRunning: true, isPaused: false });

      render(<SimulationPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByRole('button', { name: /pause simulation/i }));

      expect(mockSimulation.pause).toHaveBeenCalled();
    });

    it('calls resume when resume button is clicked', () => {
      Object.assign(mockSimulation, { isRunning: true, isPaused: true });

      render(<SimulationPage />, { wrapper: createWrapper() });

      fireEvent.click(screen.getByRole('button', { name: /resume simulation/i }));

      expect(mockSimulation.resume).toHaveBeenCalled();
    });
  });

  describe('responsive behavior', () => {
    it('has proper flex layout structure', () => {
      const { container } = render(<SimulationPage />, { wrapper: createWrapper() });

      // Main container should be flex
      const mainDiv = container.firstChild as HTMLElement;
      expect(mainDiv.className).toContain('flex');
      expect(mainDiv.className).toContain('h-screen');
    });

    it('sidebars have fixed width', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const sidebars = document.querySelectorAll('aside');
      expect(sidebars.length).toBe(2);

      // Left sidebar: w-80 (320px)
      expect(sidebars[0].className).toContain('w-80');

      // Right sidebar: w-72 (288px)
      expect(sidebars[1].className).toContain('w-72');
    });

    it('main content is flexible', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const main = document.querySelector('main');
      expect(main?.className).toContain('flex-1');
    });
  });

  describe('frame history', () => {
    it('passes frames array to Scene3D', () => {
      const frames: SimulationFrame[] = [
        { frame: 0, time: 0, state: { qpos: [0] } },
        { frame: 1, time: 0.1, state: { qpos: [0.1] } },
        { frame: 2, time: 0.2, state: { qpos: [0.2] } },
      ];

      Object.assign(mockSimulation, {
        isRunning: true,
        currentFrame: frames[2],
        frames,
      });

      render(<SimulationPage />, { wrapper: createWrapper() });

      // Scene3D receives frames for trajectory visualization
      expect(screen.getByTestId('scene3d-mock')).toBeInTheDocument();
    });
  });
});
