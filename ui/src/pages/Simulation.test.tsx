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
  // setSpeed resolves to a structured result (issue #7166), never rejects.
  setSpeed: vi.fn().mockResolvedValue({ success: true }),
};

vi.mock('@/api/client', () => ({
  useSimulation: vi.fn(() => mockSimulation),
}));

// Controllable mocks for the controls wiring (#7452). vi.hoisted so the
// vi.mock factories below can reference them.
const h = vi.hoisted(() => ({
  controls: {
    fetchCameraPresets: vi.fn(async (): Promise<{ preset: string }[]> => []),
    applyCameraPreset: vi.fn(async (preset: string) => ({
      preset,
      position: [0, 0, 0],
      target: [0, 0, 0],
      up: [0, 0, 1],
    })),
    controlRecording: vi.fn(async () => ({
      recording: true,
      frame_count: 0,
      status: 'started',
    })),
    downloadTrajectory: vi.fn(),
  },
  capabilities: {
    loadState: 'idle' as string,
    isSupported: (() => true) as (name: string) => boolean,
  },
}));

vi.mock('@/api/simulationControls', () => ({
  FALLBACK_CAMERA_PRESETS: ['side', 'front', 'top', 'follow_ball', 'follow_club'],
  fetchCameraPresets: h.controls.fetchCameraPresets,
  applyCameraPreset: h.controls.applyCameraPreset,
  controlRecording: h.controls.controlRecording,
  downloadTrajectory: h.controls.downloadTrajectory,
}));

vi.mock('@/api/useEngineCapabilities', () => ({
  useEngineCapabilities: () => ({
    capabilities: null,
    loadState: h.capabilities.loadState,
    error: null,
    fetchCapabilities: vi.fn(),
    isSupported: (name: string) => h.capabilities.isSupported(name),
    isFullySupported: () => true,
    getLevel: () => 'full',
  }),
}));

// Mock Scene3D
vi.mock('@/components/visualization/Scene3D', () => ({
  Scene3D: ({
    engine,
    frame,
    cameraCommand,
  }: {
    engine: string;
    frame: unknown;
    cameraCommand?: { preset: string; seq: number } | null;
  }) => (
    <div
      data-testid="scene3d-mock"
      data-engine={engine}
      data-has-frame={!!frame}
      data-camera-preset={cameraCommand?.preset ?? ''}
      data-camera-seq={cameraCommand?.seq ?? 0}
    >
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
      setSpeed: vi.fn().mockResolvedValue({ success: true }),
    });

    // Reset controls-wiring mocks (#7452)
    h.controls.fetchCameraPresets.mockResolvedValue([]);
    h.controls.applyCameraPreset.mockImplementation(async (preset: string) => ({
      preset,
      position: [0, 0, 0],
      target: [0, 0, 0],
      up: [0, 0, 1],
    }));
    h.controls.controlRecording.mockResolvedValue({
      recording: true,
      frame_count: 0,
      status: 'started',
    });
    h.capabilities.loadState = 'idle';
    h.capabilities.isSupported = () => true;
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

  describe('speed control slider', () => {
    beforeEach(() => {
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e: ManagedEngine) =>
          e.name === 'mujoco'
            ? { ...e, loadState: 'loaded' as const }
            : e
        ),
        selectedEngine: 'mujoco',
      }));
      Object.assign(mockSimulation, { isRunning: true });
    });

    it('renders speed factor slider when running', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });
      expect(screen.getByLabelText(/speed factor/i)).toBeInTheDocument();
    });

    it('calls setSpeed when slider value changes', () => {
      render(<SimulationPage />, { wrapper: createWrapper() });
      const slider = screen.getByLabelText(/speed factor/i);
      fireEvent.change(slider, { target: { value: '2.5' } });
      expect(mockSimulation.setSpeed).toHaveBeenCalledWith(2.5);
    });
  });

  // ── Controls wiring (#7452): camera, recording, export ────────────────
  describe('controls wiring (#7452)', () => {
    const loadMujoco = () =>
      useEngineStore.setState((state) => ({
        engines: state.engines.map((e: ManagedEngine) =>
          e.name === 'mujoco' ? { ...e, loadState: 'loaded' as const } : e
        ),
        selectedEngine: 'mujoco',
      }));

    beforeEach(() => {
      loadMujoco();
    });

    describe('camera presets', () => {
      it('POSTs the preset to the backend and drives the local 3D camera', async () => {
        render(<SimulationPage />, { wrapper: createWrapper() });

        fireEvent.click(screen.getByRole('button', { name: /top camera view/i }));

        expect(h.controls.applyCameraPreset).toHaveBeenCalledWith('top');
        await waitFor(() => {
          expect(screen.getByTestId('scene3d-mock')).toHaveAttribute(
            'data-camera-preset',
            'top'
          );
        });
      });

      it('re-applies the same preset on repeat clicks (seq bumps)', async () => {
        render(<SimulationPage />, { wrapper: createWrapper() });

        fireEvent.click(screen.getByRole('button', { name: /side camera view/i }));
        fireEvent.click(screen.getByRole('button', { name: /side camera view/i }));

        await waitFor(() => {
          expect(screen.getByTestId('scene3d-mock')).toHaveAttribute(
            'data-camera-seq',
            '2'
          );
        });
      });

      it('renders the preset list fetched from the API', async () => {
        h.controls.fetchCameraPresets.mockResolvedValue([
          { preset: 'side' },
          { preset: 'follow_ball' },
        ]);

        render(<SimulationPage />, { wrapper: createWrapper() });

        await waitFor(() => {
          expect(
            screen.queryByRole('button', { name: /top camera view/i })
          ).not.toBeInTheDocument();
        });
        expect(
          screen.getByRole('button', { name: /side camera view/i })
        ).toBeInTheDocument();
        expect(
          screen.getByRole('button', { name: /ball camera view/i })
        ).toBeInTheDocument();
      });

      it('keeps the fallback presets when the API enumeration fails', async () => {
        h.controls.fetchCameraPresets.mockRejectedValue(new Error('offline'));

        render(<SimulationPage />, { wrapper: createWrapper() });

        await waitFor(() => {
          expect(h.controls.fetchCameraPresets).toHaveBeenCalled();
        });
        expect(
          screen.getByRole('button', { name: /top camera view/i })
        ).toBeInTheDocument();
      });

      it('surfaces backend preset errors as a toast', async () => {
        h.controls.applyCameraPreset.mockRejectedValue(
          new Error('Unknown camera preset: diagonal')
        );

        render(<SimulationPage />, { wrapper: createWrapper() });
        fireEvent.click(screen.getByRole('button', { name: /front camera view/i }));

        expect(
          await screen.findByText(/unknown camera preset/i)
        ).toBeInTheDocument();
      });
    });

    describe('recording toggle', () => {
      it('start → stop round-trips through the API into the store', async () => {
        h.controls.controlRecording
          .mockResolvedValueOnce({ recording: true, frame_count: 0, status: 'started' })
          .mockResolvedValueOnce({ recording: false, frame_count: 7, status: 'stopped' });

        render(<SimulationPage />, { wrapper: createWrapper() });

        fireEvent.click(screen.getByRole('button', { name: /start recording/i }));
        await waitFor(() => {
          expect(useSimulationStore.getState().recording.status).toBe('recording');
        });
        expect(h.controls.controlRecording).toHaveBeenCalledWith('start');

        fireEvent.click(screen.getByRole('button', { name: /stop recording/i }));
        await waitFor(() => {
          expect(useSimulationStore.getState().recording).toEqual({
            status: 'saved',
            frameCount: 7,
          });
        });
        expect(h.controls.controlRecording).toHaveBeenCalledWith('stop');
        // Toast announces the saved recording with its frame count
        expect(await screen.findByText(/recording saved — 7 frames/i)).toBeInTheDocument();
      });

      it('does not flip state when the backend call fails', async () => {
        h.controls.controlRecording.mockRejectedValue(new Error('recorder offline'));

        render(<SimulationPage />, { wrapper: createWrapper() });
        fireEvent.click(screen.getByRole('button', { name: /start recording/i }));

        expect(await screen.findByText(/recorder offline/i)).toBeInTheDocument();
        expect(useSimulationStore.getState().recording.status).toBe('idle');
        expect(
          screen.getByRole('button', { name: /start recording/i })
        ).toBeInTheDocument();
      });

      it('is hidden when the engine lacks forward_sim capability', () => {
        h.capabilities.loadState = 'loaded';
        h.capabilities.isSupported = (name: string) => name !== 'forward_sim';

        render(<SimulationPage />, { wrapper: createWrapper() });

        expect(
          screen.queryByRole('button', { name: /recording/i })
        ).not.toBeInTheDocument();
        expect(
          screen.queryByRole('button', { name: /export trajectory/i })
        ).not.toBeInTheDocument();
      });
    });

    describe('trajectory export', () => {
      it('downloads the accumulated frame buffer as CSV and JSON', async () => {
        const frames: SimulationFrame[] = [
          { frame: 0, time: 0, state: { qpos: [0] } },
          { frame: 1, time: 0.1, state: { qpos: [0.1] } },
        ];
        Object.assign(mockSimulation, { frames });

        render(<SimulationPage />, { wrapper: createWrapper() });
        fireEvent.click(screen.getByRole('button', { name: /export trajectory/i }));

        expect(h.controls.downloadTrajectory).toHaveBeenCalledWith(frames, 'csv');
        expect(h.controls.downloadTrajectory).toHaveBeenCalledWith(frames, 'json');
        expect(await screen.findByText(/exported 2 frames/i)).toBeInTheDocument();
      });

      it('shows an error when there is nothing to export', async () => {
        Object.assign(mockSimulation, { frames: [] });

        render(<SimulationPage />, { wrapper: createWrapper() });
        fireEvent.click(screen.getByRole('button', { name: /export trajectory/i }));

        expect(h.controls.downloadTrajectory).not.toHaveBeenCalled();
        expect(
          await screen.findByText(/no trajectory frames to export/i)
        ).toBeInTheDocument();
      });
    });
  });

  // #6896: an unclean drop is NOT silently auto-reconnected (which would
  // restart the sim from t=0). The user is told a manual restart is required.
  describe('connection status notifications', () => {
    it('shows a restart-required notice when the socket drops mid-run', async () => {
      Object.assign(mockSimulation, { connectionStatus: 'lost' });
      render(<SimulationPage />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(
          screen.getByText(/connection lost — restart required/i),
        ).toBeInTheDocument();
      });
    });

    it('surfaces a failure notice when reconnection gives up', async () => {
      Object.assign(mockSimulation, { connectionStatus: 'failed' });
      render(<SimulationPage />, { wrapper: createWrapper() });

      await waitFor(() => {
        expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
      });
    });
  });
});
