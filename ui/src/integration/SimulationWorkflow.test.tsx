/**
 * Integration tests for the complete simulation workflow.
 *
 * These tests verify that multiple components work together correctly,
 * simulating realistic user workflows through the application.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen, fireEvent, waitFor, within } from '@testing-library/dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Use actual components - don't mock the internal ones for integration tests
// Only mock external dependencies (WebSocket, API calls)

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  // WebSocket ready state constants
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  url: string;
  readyState: number = 0;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  sentMessages: string[] = [];

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    // Simulate async connection
    setTimeout(() => this.simulateOpen(), 10);
  }

  send(data: string) {
    this.sentMessages.push(data);
    // Auto-respond with simulation frames
    const parsed = JSON.parse(data);
    if (parsed.action === 'start') {
      this.startSimulationFrames();
    }
  }

  close(code?: number) {
    this.readyState = 3;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code: code || 1000, wasClean: true }));
    }
  }

  simulateOpen() {
    this.readyState = 1;
    if (this.onopen) {
      this.onopen(new Event('open'));
    }
  }

  startSimulationFrames() {
    // Simulate sending frames - longer running simulation for testing pause/stop
    let frameNum = 0;
    const maxFrames = 50; // More frames to allow testing pause/stop
    const sendFrame = () => {
      if (this.readyState !== 1 || frameNum >= maxFrames) {
        if (this.onmessage && this.readyState === 1) {
          this.onmessage(new MessageEvent('message', {
            data: JSON.stringify({ status: 'complete' }),
          }));
        }
        return;
      }

      if (this.onmessage) {
        this.onmessage(new MessageEvent('message', {
          data: JSON.stringify({
            frame: frameNum,
            time: frameNum * 0.02,
            state: { qpos: [0.1 * frameNum] },
            analysis: { joint_angles: [0.5, 0.3, 0.2, 0.1] },
          }),
        }));
      }
      frameNum++;
      setTimeout(sendFrame, 50); // Slower frame rate
    };
    setTimeout(sendFrame, 20);
  }

  static reset() {
    MockWebSocket.instances = [];
  }

  static getLastInstance(): MockWebSocket | undefined {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }
}

// Mock react-three-fiber since we can't render WebGL in tests
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="canvas-mock">{children}</div>
  ),
  useFrame: vi.fn(),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => null,
  Grid: () => null,
  Environment: () => null,
  Line: () => null,
}));

vi.mock('three', () => ({
  Group: class {},
  Mesh: class {},
}));

// Mock fetch for API calls
const mockEngines = [
  { name: 'mujoco', available: true, loaded: true, capabilities: ['rigid_body', 'contact'] },
  { name: 'drake', available: true, loaded: false, capabilities: ['rigid_body', 'optimization'] },
  { name: 'pinocchio', available: false, loaded: false, capabilities: ['rigid_body'] },
];

// Import after mocks
import { SimulationPage } from '@/pages/Simulation';

const createTestWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('Simulation Workflow Integration', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    MockWebSocket.reset();
    vi.stubGlobal('WebSocket', MockWebSocket);

    // Mock fetch for engines API
    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/engines')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ engines: mockEngines }),
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  describe('complete simulation flow', () => {
    it('allows user to select engine, start, and view simulation', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      // Wait for engines to load
      await waitFor(() => {
        expect(screen.getByText('mujoco')).toBeInTheDocument();
      });

      // Verify initial state
      expect(screen.getByText('Ready')).toBeInTheDocument();
      expect(screen.getByText(/start simulation to view live data/i)).toBeInTheDocument();

      // Click start button
      const startButton = screen.getByRole('button', { name: /start simulation/i });
      fireEvent.click(startButton);

      // Advance timers to allow WebSocket to connect
      await vi.advanceTimersByTimeAsync(50);

      // Verify simulation starts
      await waitFor(() => {
        const ws = MockWebSocket.getLastInstance();
        expect(ws).toBeDefined();
        expect(ws!.url).toContain('mujoco');
      });

      // Advance timers for frames to be received
      await vi.advanceTimersByTimeAsync(200);

      // Verify we received frames - check for Frame text pattern
      await waitFor(() => {
        // The status should show frame count
        expect(screen.getByText(/Frame \d+/)).toBeInTheDocument();
      });
    });

    it('allows switching engines before starting simulation', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      // Wait for engines to load
      await waitFor(() => {
        expect(screen.getByText('drake')).toBeInTheDocument();
      });

      // Select drake engine
      const drakeRadio = screen.getByRole('radio', { name: /drake/i });
      fireEvent.click(drakeRadio);

      // Start simulation
      const startButton = screen.getByRole('button', { name: /start simulation/i });
      fireEvent.click(startButton);

      await vi.advanceTimersByTimeAsync(50);

      // Verify WebSocket connects to drake endpoint
      const ws = MockWebSocket.getLastInstance();
      expect(ws!.url).toContain('drake');
    });

    it('disables engine switching while simulation is running', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      await waitFor(() => {
        expect(screen.getByText('mujoco')).toBeInTheDocument();
      });

      // Start simulation
      fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));

      await vi.advanceTimersByTimeAsync(50);

      // Engine radios should be disabled
      await waitFor(() => {
        const radios = screen.getAllByRole('radio');
        radios.forEach((radio) => {
          expect(radio).toBeDisabled();
        });
      });
    });
  });

  describe('pause and resume workflow', () => {
    it('allows pausing and resuming simulation', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      await waitFor(() => {
        expect(screen.getByText('mujoco')).toBeInTheDocument();
      });

      // Start simulation
      fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));

      // Wait for WebSocket to connect and simulation to start
      await vi.advanceTimersByTimeAsync(50);

      // Verify simulation is running
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /pause simulation/i })).toBeInTheDocument();
      }, { timeout: 500 });

      // Click pause
      const pauseButton = screen.getByRole('button', { name: /pause simulation/i });
      fireEvent.click(pauseButton);

      // Verify pause command was sent to WebSocket
      const ws = MockWebSocket.getLastInstance();
      expect(ws).toBeDefined();
      const pauseMessage = ws!.sentMessages.find((m) => {
        try {
          return JSON.parse(m).action === 'pause';
        } catch {
          return false;
        }
      });
      expect(pauseMessage).toBeDefined();

      // Resume button should appear after pause
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /resume simulation/i })).toBeInTheDocument();
      }, { timeout: 500 });

      // Click resume
      fireEvent.click(screen.getByRole('button', { name: /resume simulation/i }));

      // Verify resume command was sent
      const resumeMessage = ws!.sentMessages.find((m) => {
        try {
          return JSON.parse(m).action === 'resume';
        } catch {
          return false;
        }
      });
      expect(resumeMessage).toBeDefined();
    });
  });

  describe('stop workflow', () => {
    it('allows stopping simulation and returning to ready state', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      await waitFor(() => {
        expect(screen.getByText('mujoco')).toBeInTheDocument();
      });

      // Start simulation
      fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));

      // Wait for WebSocket to connect
      await vi.advanceTimersByTimeAsync(50);

      // Stop button should be visible
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /stop simulation/i })).toBeInTheDocument();
      }, { timeout: 500 });

      // Click stop
      const stopButton = screen.getByRole('button', { name: /stop simulation/i });
      fireEvent.click(stopButton);

      // Verify stop command was sent
      const ws = MockWebSocket.getLastInstance();
      expect(ws).toBeDefined();
      const stopMessage = ws!.sentMessages.find((m) => {
        try {
          return JSON.parse(m).action === 'stop';
        } catch {
          return false;
        }
      });
      expect(stopMessage).toBeDefined();

      // Wait for state to update
      await vi.advanceTimersByTimeAsync(100);

      // Should return to ready state
      await waitFor(() => {
        expect(screen.getByText('Ready')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /start simulation/i })).toBeInTheDocument();
      }, { timeout: 1000 });
    });
  });

  describe('error handling', () => {
    it('handles engine loading failure gracefully', async () => {
      // Override fetch to fail
      global.fetch = vi.fn().mockRejectedValue(new Error('Network error'));

      render(<SimulationPage />, { wrapper: createTestWrapper() });

      // Should show error state
      await waitFor(() => {
        expect(screen.getByRole('alert')).toBeInTheDocument();
      });
    });

    it('shows unavailable engines as disabled', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      await waitFor(() => {
        expect(screen.getByText('pinocchio')).toBeInTheDocument();
      });

      // Pinocchio should be disabled (not installed)
      const pinocchioRadio = screen.getByRole('radio', { name: /pinocchio.*not installed/i });
      expect(pinocchioRadio).toBeDisabled();
    });
  });

  describe('live analysis display', () => {
    it('displays simulation state during running simulation', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      await waitFor(() => {
        expect(screen.getByText('mujoco')).toBeInTheDocument();
      });

      // Start simulation
      fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));

      await vi.advanceTimersByTimeAsync(200);

      // Live analysis panel should show state data
      await waitFor(() => {
        expect(screen.getByText('Simulation State')).toBeInTheDocument();
        // JSON state should be displayed
        expect(screen.getByText(/"qpos"/)).toBeInTheDocument();
      });
    });
  });

  describe('3D visualization', () => {
    it('renders 3D scene component', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      // Scene3D should be rendered (as mocked Canvas)
      expect(screen.getByTestId('canvas-mock')).toBeInTheDocument();
    });

    it('3D scene has accessibility attributes', async () => {
      render(<SimulationPage />, { wrapper: createTestWrapper() });

      const scene = screen.getByRole('img', {
        name: /3D golf swing simulation visualization/i,
      });
      expect(scene).toBeInTheDocument();
      expect(scene).toHaveAttribute('tabIndex', '0');
    });
  });
});

describe('Multi-session workflow', () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    MockWebSocket.reset();
    vi.stubGlobal('WebSocket', MockWebSocket);

    global.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/engines')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ engines: mockEngines }),
        });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it('creates new WebSocket connection on each simulation start', async () => {
    render(<SimulationPage />, { wrapper: createTestWrapper() });

    await waitFor(() => {
      expect(screen.getByText('mujoco')).toBeInTheDocument();
    });

    // First simulation - start it
    fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));

    // Wait for WebSocket to connect
    await vi.advanceTimersByTimeAsync(50);

    const firstWsCount = MockWebSocket.instances.length;
    expect(firstWsCount).toBeGreaterThan(0);

    // Stop the simulation manually
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /stop simulation/i })).toBeInTheDocument();
    }, { timeout: 500 });

    fireEvent.click(screen.getByRole('button', { name: /stop simulation/i }));

    // Wait for state to settle
    await vi.advanceTimersByTimeAsync(100);

    // Wait for ready state
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /start simulation/i })).toBeInTheDocument();
    }, { timeout: 1000 });

    // Start second simulation
    fireEvent.click(screen.getByRole('button', { name: /start simulation/i }));
    await vi.advanceTimersByTimeAsync(50);

    // New WebSocket should be created
    expect(MockWebSocket.instances.length).toBeGreaterThan(firstWsCount);
  });
});
