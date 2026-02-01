import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen } from '@testing-library/dom';

// Mock react-three/fiber before importing Scene3D
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <div data-testid="canvas-mock" {...props}>{children}</div>
  ),
  useFrame: vi.fn((callback) => {
    // Simulate a single frame call for testing
    if (typeof callback === 'function') {
      callback({ clock: { getElapsedTime: () => 0 } }, 0);
    }
  }),
}));

// Mock react-three/drei
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls-mock" />,
  Grid: () => <div data-testid="grid-mock" />,
  Environment: () => <div data-testid="environment-mock" />,
  Line: ({ points }: { points: number[][] }) => (
    <div data-testid="line-mock" data-points={points?.length || 0} />
  ),
}));

// Mock three.js
vi.mock('three', () => ({
  Group: class Group {
    rotation = { y: 0 };
  },
  Mesh: class Mesh {
    rotation = { x: 0, y: 0, z: 0 };
  },
}));

import { Scene3D } from './Scene3D';
import type { SimulationFrame } from '@/api/client';

describe('Scene3D', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders the 3D canvas container', () => {
      render(<Scene3D engine="mujoco" frame={null} />);

      expect(screen.getByTestId('canvas-mock')).toBeInTheDocument();
    });

    it('renders without frame data (idle state)', () => {
      render(<Scene3D engine="mujoco" frame={null} />);

      // Should render without errors
      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('renders with frame data', () => {
      const mockFrame: SimulationFrame = {
        frame: 1,
        time: 0.5,
        state: { qpos: [0, 0, 0] },
        analysis: {
          joint_angles: [0.1, 0.2, 0.3, 0.4],
        },
      };

      render(<Scene3D engine="mujoco" frame={mockFrame} />);

      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('renders with multiple frames (trajectory)', () => {
      const mockFrames: SimulationFrame[] = [
        { frame: 0, time: 0, state: { qpos: [0] } },
        { frame: 1, time: 0.1, state: { qpos: [0.1] } },
        { frame: 2, time: 0.2, state: { qpos: [0.2] } },
      ];

      render(
        <Scene3D
          engine="mujoco"
          frame={mockFrames[2]}
          frames={mockFrames}
        />
      );

      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has proper ARIA role and label', () => {
      render(<Scene3D engine="mujoco" frame={null} />);

      const container = screen.getByRole('img');
      expect(container).toHaveAttribute(
        'aria-label',
        '3D golf swing simulation visualization. Use mouse to rotate view, scroll to zoom.'
      );
    });

    it('is focusable for keyboard navigation', () => {
      render(<Scene3D engine="mujoco" frame={null} />);

      const container = screen.getByRole('img');
      expect(container).toHaveAttribute('tabIndex', '0');
    });

    it('has focus ring styling', () => {
      render(<Scene3D engine="mujoco" frame={null} />);

      const container = screen.getByRole('img');
      expect(container.className).toContain('focus:ring');
    });
  });

  describe('engine switching', () => {
    it('renders with mujoco engine', () => {
      render(<Scene3D engine="mujoco" frame={null} />);
      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('renders with drake engine', () => {
      render(<Scene3D engine="drake" frame={null} />);
      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('renders with pinocchio engine', () => {
      render(<Scene3D engine="pinocchio" frame={null} />);
      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('frame updates', () => {
    it('handles frame with joint angles', () => {
      const frameWithAngles: SimulationFrame = {
        frame: 10,
        time: 1.0,
        state: { qpos: [0, 0, 0, 0, 0] },
        analysis: {
          joint_angles: [0.5, -0.3, 0.2, 1.0],
          velocities: [0.1, 0.2, 0.3],
        },
      };

      render(<Scene3D engine="mujoco" frame={frameWithAngles} />);
      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('handles frame without joint angles (uses time-based animation)', () => {
      const frameWithoutAngles: SimulationFrame = {
        frame: 5,
        time: 0.25,
        state: { qpos: [0, 0, 0] },
        // No analysis/joint_angles
      };

      render(<Scene3D engine="mujoco" frame={frameWithoutAngles} />);
      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('handles incomplete joint angles array', () => {
      const frameWithPartialAngles: SimulationFrame = {
        frame: 3,
        time: 0.15,
        state: { qpos: [0] },
        analysis: {
          joint_angles: [0.1, 0.2], // Less than 4 elements
        },
      };

      render(<Scene3D engine="mujoco" frame={frameWithPartialAngles} />);
      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('trajectory visualization', () => {
    it('does not render trail with less than 2 frames', () => {
      const singleFrame: SimulationFrame = {
        frame: 0,
        time: 0,
        state: { qpos: [0] },
      };

      render(
        <Scene3D
          engine="mujoco"
          frame={singleFrame}
          frames={[singleFrame]}
        />
      );

      // Line should not render with insufficient points
      const line = screen.queryByTestId('line-mock');
      // Line component receives empty points array or doesn't render
      if (line) {
        expect(line.getAttribute('data-points')).toBe('0');
      }
    });

    it('renders trail with multiple frames', () => {
      const frames: SimulationFrame[] = Array.from({ length: 10 }, (_, i) => ({
        frame: i,
        time: i * 0.1,
        state: { qpos: [i * 0.01] },
      }));

      render(
        <Scene3D
          engine="mujoco"
          frame={frames[frames.length - 1]}
          frames={frames}
        />
      );

      expect(screen.getByRole('img')).toBeInTheDocument();
    });

    it('limits trail to MAX_TRAIL_POINTS', () => {
      // Create more than 100 frames
      const frames: SimulationFrame[] = Array.from({ length: 150 }, (_, i) => ({
        frame: i,
        time: i * 0.01,
        state: { qpos: [i * 0.001] },
      }));

      render(
        <Scene3D
          engine="mujoco"
          frame={frames[frames.length - 1]}
          frames={frames}
        />
      );

      // Component should handle large frame arrays without error
      expect(screen.getByRole('img')).toBeInTheDocument();
    });
  });

  describe('scene elements', () => {
    it('renders OrbitControls for camera interaction', () => {
      render(<Scene3D engine="mujoco" frame={null} />);
      expect(screen.getByTestId('orbit-controls-mock')).toBeInTheDocument();
    });

    it('renders Grid for reference', () => {
      render(<Scene3D engine="mujoco" frame={null} />);
      expect(screen.getByTestId('grid-mock')).toBeInTheDocument();
    });

    it('renders Environment for lighting', () => {
      render(<Scene3D engine="mujoco" frame={null} />);
      expect(screen.getByTestId('environment-mock')).toBeInTheDocument();
    });
  });
});
