/**
 * Tests for MocapSkeleton3D component.
 *
 * See issue #8406
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen } from '@testing-library/dom';

// Mock react-three/fiber before importing the component
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <div data-testid="canvas-mock" {...props}>{children}</div>
  ),
  useFrame: vi.fn(),
}));

// Mock react-three/drei
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls-mock" />,
  Grid: () => <div data-testid="grid-mock" />,
  Line: ({ points }: { points: number[][] }) => (
    <div data-testid="bone-line-mock" data-points={points?.length || 0} />
  ),
}));

import { MocapSkeleton3D } from './MocapSkeleton3D';
import type { MocapJoint } from './MocapSkeleton3D';

const SAMPLE_JOINTS: MocapJoint[] = [
  { name: 'hips', position: [0, 0.9, 0.1], confidence: 0.95, parent: null },
  { name: 'spine', position: [0, 1.2, 0.1], confidence: 0.9, parent: 'hips' },
  { name: 'head', position: [0, 1.6, 0.12], confidence: 0.85, parent: 'spine' },
  {
    name: 'left_shoulder',
    position: [-0.2, 1.4, 0.1],
    confidence: 0.8,
    parent: 'spine',
  },
];

describe('MocapSkeleton3D', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the Canvas with orbit controls and grid', () => {
    render(<MocapSkeleton3D joints={SAMPLE_JOINTS} />);

    expect(screen.getByTestId('mocap-skeleton-3d')).toBeInTheDocument();
    expect(screen.getByTestId('canvas-mock')).toBeInTheDocument();
    expect(screen.getByTestId('orbit-controls-mock')).toBeInTheDocument();
    expect(screen.getByTestId('grid-mock')).toBeInTheDocument();
  });

  it('renders one sphere mesh per joint', () => {
    const { container } = render(<MocapSkeleton3D joints={SAMPLE_JOINTS} />);

    // <mesh> elements come out of the mocked Canvas as literal DOM tags.
    const meshes = container.querySelectorAll('mesh');
    expect(meshes).toHaveLength(SAMPLE_JOINTS.length);
  });

  it('renders one bone segment per parented joint', () => {
    render(<MocapSkeleton3D joints={SAMPLE_JOINTS} />);

    // 3 joints have a parent -> 3 bones, each a 2-point line
    const bones = screen.getAllByTestId('bone-line-mock');
    expect(bones).toHaveLength(3);
    for (const bone of bones) {
      expect(bone).toHaveAttribute('data-points', '2');
    }
  });

  it('skips bones whose parent joint is missing from the frame', () => {
    const joints: MocapJoint[] = [
      { name: 'hand', position: [0, 1, 0], confidence: 1, parent: 'wrist' },
    ];
    render(<MocapSkeleton3D joints={joints} />);

    expect(screen.queryByTestId('bone-line-mock')).toBeNull();
  });

  it('renders an empty scene without crashing when no joints are given', () => {
    const { container } = render(<MocapSkeleton3D joints={[]} />);

    expect(screen.getByTestId('mocap-skeleton-3d')).toBeInTheDocument();
    expect(container.querySelectorAll('mesh')).toHaveLength(0);
  });

  it('treats missing z components as zero', () => {
    const joints: MocapJoint[] = [
      { name: 'a', position: [0.1, 0.2], confidence: 1, parent: null },
      { name: 'b', position: [0.2, 0.4], confidence: 1, parent: 'a' },
    ];
    const { container } = render(<MocapSkeleton3D joints={joints} />);

    expect(container.querySelectorAll('mesh')).toHaveLength(2);
    expect(screen.getAllByTestId('bone-line-mock')).toHaveLength(1);
  });
});
