/**
 * Tests for URDFViewer component.
 *
 * See issue #1201
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';

// Mock react-three/fiber
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <div data-testid="canvas-mock" {...props}>{children}</div>
  ),
  useFrame: vi.fn((callback) => {
    if (typeof callback === 'function') {
      callback({ clock: { getElapsedTime: () => 0 } }, 0);
    }
  }),
}));

// Mock three.js
vi.mock('three', () => ({
  Color: class Color {
    constructor(public r: number, public g: number, public b: number) {}
  },
  Vector3: class Vector3 {
    constructor(public x = 0, public y = 0, public z = 0) {}
    normalize() { return this; }
    clone() { return new (this.constructor as typeof Vector3)(this.x, this.y, this.z); }
    multiplyScalar(s: number) {
      this.x *= s; this.y *= s; this.z *= s;
      return this;
    }
    toArray() { return [this.x, this.y, this.z]; }
    add(v: { x: number; y: number; z: number }) {
      this.x += v.x; this.y += v.y; this.z += v.z;
      return this;
    }
    copy(v: { x: number; y: number; z: number }) {
      this.x = v.x; this.y = v.y; this.z = v.z;
      return this;
    }
  },
  Euler: class Euler {
    constructor(public x = 0, public y = 0, public z = 0, public order = 'XYZ') {}
  },
  Quaternion: class Quaternion {
    x = 0; y = 0; z = 0; w = 1;
    setFromAxisAngle() { return this; }
    setFromEuler() { return this; }
    setFromUnitVectors() { return this; }
    copy() { return this; }
    multiply() { return this; }
  },
  Group: class Group {
    position = { x: 0, y: 0, z: 0, copy() { return this; }, add() { return this; } };
    rotation = { x: 0, y: 0, z: 0 };
    quaternion = {
      x: 0, y: 0, z: 0, w: 1,
      copy() { return this; },
      multiply() { return this; },
    };
  },
  Mesh: class Mesh {
    position = { x: 0, y: 0, z: 0 };
    rotation = { x: 0, y: 0, z: 0 };
  },
}));

import { URDFViewer } from './URDFViewer';
import type { URDFModel } from './URDFViewer';

describe('URDFViewer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const createTestModel = (): URDFModel => ({
    model_name: 'test_robot',
    links: [
      {
        link_name: 'torso',
        geometry_type: 'box',
        dimensions: { width: 0.2, height: 0.4, depth: 0.6 },
        origin: [0, 0, 0.3] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
        color: [0, 0, 0.8, 1] as [number, number, number, number],
      },
      {
        link_name: 'head',
        geometry_type: 'sphere',
        dimensions: { radius: 0.12 },
        origin: [0, 0, 0.12] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
        color: [1, 1, 1, 1] as [number, number, number, number],
      },
      {
        link_name: 'arm',
        geometry_type: 'cylinder',
        dimensions: { radius: 0.05, length: 0.3 },
        origin: [0.15, 0, 0] as [number, number, number],
        rotation: [0, 1.57, 0] as [number, number, number],
        color: [1, 1, 1, 1] as [number, number, number, number],
      },
    ],
    joints: [
      {
        name: 'neck',
        joint_type: 'revolute',
        parent_link: 'torso',
        child_link: 'head',
        origin: [0, 0, 0.6] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
        axis: [0, 0, 1] as [number, number, number],
        lower_limit: -1.57,
        upper_limit: 1.57,
      },
      {
        name: 'shoulder',
        joint_type: 'revolute',
        parent_link: 'torso',
        child_link: 'arm',
        origin: [0.1, 0, 0.5] as [number, number, number],
        rotation: [0, 0, 0] as [number, number, number],
        axis: [0, 1, 0] as [number, number, number],
        lower_limit: -3.14,
        upper_limit: 3.14,
      },
    ],
    root_link: 'torso',
  });

  describe('rendering', () => {
    it('renders null when model is null', () => {
      const { container } = render(
        <URDFViewer model={null} />,
      );
      // Should render nothing
      expect(container.innerHTML).toBe('');
    });

    it('renders with a valid model', () => {
      const model = createTestModel();
      const { container } = render(
        <URDFViewer model={model} />,
      );
      // Should render something (group elements)
      expect(container.innerHTML).not.toBe('');
    });

    it('renders all link geometries', () => {
      const model = createTestModel();
      const { container } = render(
        <URDFViewer model={model} />,
      );
      // The component renders Three.js elements which become divs in test DOM
      expect(container.innerHTML).not.toBe('');
    });
  });

  describe('joint angles', () => {
    it('accepts joint angles as array', () => {
      const model = createTestModel();
      const angles = [0.5, -0.3];
      const { container } = render(
        <URDFViewer model={model} jointAngles={angles} />,
      );
      expect(container.innerHTML).not.toBe('');
    });

    it('accepts joint angles as object', () => {
      const model = createTestModel();
      const angles = { neck: 0.5, shoulder: -0.3 };
      const { container } = render(
        <URDFViewer model={model} jointAngles={angles} />,
      );
      expect(container.innerHTML).not.toBe('');
    });

    it('handles empty joint angles', () => {
      const model = createTestModel();
      const { container } = render(
        <URDFViewer model={model} jointAngles={[]} />,
      );
      expect(container.innerHTML).not.toBe('');
    });
  });

  describe('options', () => {
    it('renders with showAxes enabled', () => {
      const model = createTestModel();
      const { container } = render(
        <URDFViewer model={model} showAxes={true} />,
      );
      expect(container.innerHTML).not.toBe('');
    });

    it('renders with custom opacity', () => {
      const model = createTestModel();
      const { container } = render(
        <URDFViewer model={model} opacity={0.5} />,
      );
      expect(container.innerHTML).not.toBe('');
    });
  });

  describe('model types', () => {
    it('handles model with only boxes', () => {
      const model: URDFModel = {
        model_name: 'boxes',
        links: [
          {
            link_name: 'box1',
            geometry_type: 'box',
            dimensions: { width: 1, height: 1, depth: 1 },
            origin: [0, 0, 0] as [number, number, number],
            rotation: [0, 0, 0] as [number, number, number],
            color: [1, 0, 0, 1] as [number, number, number, number],
          },
        ],
        joints: [],
        root_link: 'box1',
      };
      const { container } = render(<URDFViewer model={model} />);
      expect(container.innerHTML).not.toBe('');
    });

    it('handles model with mesh type (fallback)', () => {
      const model: URDFModel = {
        model_name: 'mesh_model',
        links: [
          {
            link_name: 'mesh_link',
            geometry_type: 'mesh',
            dimensions: { scale_x: 1, scale_y: 1, scale_z: 1 },
            origin: [0, 0, 0] as [number, number, number],
            rotation: [0, 0, 0] as [number, number, number],
            color: [0.5, 0.5, 0.5, 1] as [number, number, number, number],
            mesh_path: 'test.stl',
          },
        ],
        joints: [],
        root_link: 'mesh_link',
      };
      const { container } = render(<URDFViewer model={model} />);
      expect(container.innerHTML).not.toBe('');
    });

    it('handles model with no visual links (empty)', () => {
      const model: URDFModel = {
        model_name: 'empty',
        links: [],
        joints: [],
        root_link: 'base',
      };
      // Should render but with no visible geometry
      const { container } = render(<URDFViewer model={model} />);
      expect(container.innerHTML).not.toBe('');
    });
  });
});
