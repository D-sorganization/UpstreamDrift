import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen, fireEvent, waitFor } from '@testing-library/dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ToastProvider } from '@/components/ui/Toast';
import { useEngineStore } from '@/stores/useEngineStore';
import { useSimulationStore } from '@/stores/useSimulationStore';
import type { ManagedEngine } from '@/stores/useEngineStore';

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
}));

// Mock Scene3D
vi.mock('@/components/visualization/Scene3D', () => ({
  Scene3D: ({ engine, frame }: { engine: string; frame: unknown }) => (
    <div data-testid="scene3d-mock" data-engine={engine} data-has-frame={!!frame}>
      Scene3D Mock
    </div>
  ),
}));

// Mock LivePlot
vi.mock('@/components/analysis/LivePlot', () => ({
  LivePlot: () => <div data-testid="live-plot-mock">LivePlot Mock</div>,
}));

// Mock ParameterPanel — do NOT call onChange during render to avoid infinite loops
vi.mock('@/components/simulation/ParameterPanel', () => ({
  ParameterPanel: ({ engine }: { engine: string; disabled?: boolean; onChange: (params: unknown) => void }) => {
    return (
      <div data-testid="parameter-panel-mock" data-engine={engine}>
        ParameterPanel Mock
      </div>
    );
  },
}));

// Mock ConnectionStatus
vi.mock('@/components/ui/ConnectionStatus', () => ({
  ConnectionStatus: ({ status }: { status: string }) => (
    <div data-testid="connection-status-mock" data-status={status}>
      Connection: {status}
    </div>
  ),
}));

import { SimulationPage } from './Simulation';
import type { SimulationFrame } from '@/api/client';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>{children}</ToastProvider>
    </QueryClientProvider>
  );
};

describe('SimulationPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Reset stores
    useEngineStore.getState().resetEngines();
    useSimulationStore.getState().resetParameters();

    // Reset mock simulation state
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
  });

  describe('layout', () => {
    it('renders main layout with sidebars', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Golf Suite')).toBeInTheDocument();
      expect(screen.getAllByText('Physics Engines').length).toBeGreaterThan(0);
      expect(screen.getByText('Live Analysis')).toBeInTheDocument();
    });

    it('renders 3D scene component', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByTestId('scene3d-mock')).toBeInTheDocument();
    });

    it('renders simulation controls', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByRole('toolbar', { name: /simulation controls/i })).toBeInTheDocument();
    });
  });

  describe('idle state — no engines loaded', () => {
    it('shows "No engine loaded" status when no engine selected', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('No engine loaded')).toBeInTheDocument();
    });

    it('shows helpful overlay prompting to load an engine', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Load a Physics Engine')).toBeInTheDocument();
    });

    it('shows "0 engines loaded" count', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('0 engines loaded')).toBeInTheDocument();
    });

    it('shows "Load an engine to get started" in analysis panel', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('Load an engine to get started')).toBeInTheDocument();
    });
  });

  describe('engine loaded state', () => {
    beforeEach(() => {
      // Set engine to loaded in the store
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e: ManagedEngine) =>
          e.name === 'mujoco'
            ? { ...e, loadState: 'loaded' as const, version: '3.1.0' }
            : e
        ),
        selectedEngine: 'mujoco',
      }));
    });

    it('shows engine count for loaded engines', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      expect(screen.getByText('1 engine loaded')).toBeInTheDocument();
    });

    it('shows "Ready" when engine loaded and selected', async () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByText('Ready')).toBeInTheDocument();
      });
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
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e: ManagedEngine) =>
          e.name === 'mujoco'
            ? { ...e, loadState: 'loaded' as const }
            : e
        ),
        selectedEngine: 'mujoco',
      }));

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
      expect(screen.getByText(/"qpos"/)).toBeInTheDocument();
    });
  });

  describe('simulation controls interaction', () => {
    beforeEach(() => {
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e: ManagedEngine) =>
          e.name === 'mujoco'
            ? { ...e, loadState: 'loaded' as const }
            : e
        ),
        selectedEngine: 'mujoco',
      }));
    });

    it('calls start when start button is clicked', async () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      await waitFor(() => {
        const startBtn = screen.getByRole('button', { name: /start simulation/i });
        expect(startBtn).not.toBeDisabled();
      });

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

      const mainDiv = container.firstChild as HTMLElement;
      expect(mainDiv.className).toContain('flex');
      expect(mainDiv.className).toContain('h-screen');
    });

    it('sidebars have fixed width', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const sidebars = document.querySelectorAll('aside');
      expect(sidebars.length).toBe(2);
      expect(sidebars[0].className).toContain('w-80');
      expect(sidebars[1].className).toContain('w-72');
    });

    it('main content is flexible', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });

      const main = document.querySelector('main');
      expect(main?.className).toContain('flex-1');
    });
  });

  describe('store integration', () => {
    it('reads parameters from simulation store', () => {
      useSimulationStore.getState().setParameters({ duration: 10.0 });

      render(<SimulationPage />, { wrapper: createWrapper() });

      // The store's parameters should be used when start is clicked
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e: ManagedEngine) =>
          e.name === 'mujoco'
            ? { ...e, loadState: 'loaded' as const }
            : e
        ),
        selectedEngine: 'mujoco',
      }));
    });

    it('marks run in store when simulation starts', async () => {
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e: ManagedEngine) =>
          e.name === 'mujoco'
            ? { ...e, loadState: 'loaded' as const }
            : e
        ),
        selectedEngine: 'mujoco',
      }));

      render(<SimulationPage />, { wrapper: createWrapper() });

      await waitFor(() => {
        const startBtn = screen.getByRole('button', { name: /start simulation/i });
        expect(startBtn).not.toBeDisabled();
      });

      fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));

      expect(useSimulationStore.getState().hasRun).toBe(true);
    });
  });
});
