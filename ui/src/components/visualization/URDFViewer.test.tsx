/**
 * Tests for URDFViewer component.
 *
 * See issue #1201
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';

// Mock react-three/drei (useGLTF backs the glTF mesh path, #8406)
const { mockUseGLTF } = vi.hoisted(() => ({
  mockUseGLTF: vi.fn(() => ({
    scene: { clone: vi.fn(() => ({ isObject3D: true })) },
  })),
}));
vi.mock('@react-three/drei', () => ({
  useGLTF: mockUseGLTF,
}));

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
import { isGltfMeshPath, meshAssetUrl } from './urdfMeshAsset';

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

    it('handles model with non-glTF mesh type (primitive fallback, no loader)', () => {
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
      // .stl is not browser-loadable; the glTF loader must not be invoked.
      expect(mockUseGLTF).not.toHaveBeenCalled();
      expect(container.querySelector('primitive')).toBeNull();
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

  describe('glTF meshes (#8406)', () => {
    const gltfModel = (meshPath: string): URDFModel => ({
      model_name: 'gltf_model',
      links: [
        {
          link_name: 'club_head',
          geometry_type: 'mesh',
          dimensions: { scale_x: 2, scale_y: 2, scale_z: 2 },
          origin: [0, 0, 0.1] as [number, number, number],
          rotation: [0, 0, 0] as [number, number, number],
          color: [0.5, 0.5, 0.5, 1] as [number, number, number, number],
          mesh_path: meshPath,
        },
      ],
      joints: [],
      root_link: 'club_head',
    });

    it('loads .glb mesh paths through useGLTF against the mesh-asset endpoint', () => {
      const { container } = render(
        <URDFViewer model={gltfModel('meshes/driver.glb')} />,
      );

      expect(mockUseGLTF).toHaveBeenCalledTimes(1);
      expect(mockUseGLTF).toHaveBeenCalledWith(
        expect.stringContaining('/api/models/mesh-asset?path=meshes%2Fdriver.glb'),
      );
      // The loaded scene is mounted via <primitive>
      expect(container.querySelector('primitive')).not.toBeNull();
    });

    it('loads .gltf mesh paths through useGLTF', () => {
      render(<URDFViewer model={gltfModel('meshes/scene.gltf')} />);

      expect(mockUseGLTF).toHaveBeenCalledWith(
        expect.stringContaining('path=meshes%2Fscene.gltf'),
      );
    });

    it('does not invoke useGLTF for primitive geometry', () => {
      const model = createTestModel();
      render(<URDFViewer model={model} />);
      expect(mockUseGLTF).not.toHaveBeenCalled();
    });

    it('isGltfMeshPath detects loadable mesh paths', () => {
      expect(isGltfMeshPath('meshes/a.glb')).toBe(true);
      expect(isGltfMeshPath('meshes/a.GLTF')).toBe(true);
      expect(isGltfMeshPath('meshes/a.glb?v=2')).toBe(true);
      expect(isGltfMeshPath('meshes/a.stl')).toBe(false);
      expect(isGltfMeshPath('meshes/a.dae')).toBe(false);
      expect(isGltfMeshPath('')).toBe(false);
      expect(isGltfMeshPath(null)).toBe(false);
      expect(isGltfMeshPath(undefined)).toBe(false);
    });

    it('meshAssetUrl encodes the mesh path and rejects empty input', () => {
      expect(meshAssetUrl('meshes/club head.glb')).toContain(
        '/api/models/mesh-asset?path=meshes%2Fclub%20head.glb',
      );
      expect(() => meshAssetUrl('')).toThrow();
    });
  });
});
