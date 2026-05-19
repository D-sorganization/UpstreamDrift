import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { JointManipulator } from '../components/model-explorer/JointManipulator';

// Mock react-three/fiber
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children, ...props }: { children: React.ReactNode; [key: string]: unknown }) => (
    <div data-testid="canvas-mock" {...props}>
      {children}
    </div>
  ),
}));

// Mock react-three/drei
vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls-mock" />,
  Grid: () => <div data-testid="grid-mock" />,
  Environment: () => <div data-testid="environment-mock" />,
}));

// Mock useURDFModel API hook
vi.mock('@/api/useURDFModel', () => ({
  useURDFModel: () => ({ model: null, loading: false, error: null }),
}));

import { ModelExplorerPage } from './ModelExplorer';

describe('ModelExplorer Page UI & Frankenstein Mode', () => {
  const mockModelsResponse = {
    models: [
      { name: 'source_robot', format: 'urdf', path: 'source.urdf' },
      { name: 'target_robot', format: 'urdf', path: 'target.urdf' },
    ],
  };

  const mockSourceResponse = {
    model_name: 'source_robot',
    tree: [
      {
        id: 'src_root',
        name: 'src_root',
        node_type: 'root',
        parent_id: null,
        children: ['src_joint_1'],
        properties: { mass: 10 },
      },
      {
        id: 'src_joint_1',
        name: 'src_joint_1',
        node_type: 'joint',
        parent_id: 'src_root',
        children: [],
        properties: { joint_type: 'revolute' },
      },
    ],
    joint_count: 1,
    link_count: 1,
    model_format: 'urdf',
    file_path: 'source.urdf',
  };

  const mockTargetResponse = {
    model_name: 'target_robot',
    tree: [
      {
        id: 'tgt_root',
        name: 'tgt_root',
        node_type: 'root',
        parent_id: null,
        children: [],
        properties: { mass: 12 },
      },
    ],
    joint_count: 0,
    link_count: 1,
    model_format: 'urdf',
    file_path: 'target.urdf',
  };

  const mockFetch = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockReset();
    vi.stubGlobal('fetch', mockFetch);

    // Default mock behavior for fetch
    mockFetch.mockImplementation(async (url: string) => {
      if (url.includes('/api/models')) {
        return {
          ok: true,
          json: async () => mockModelsResponse,
        };
      }
      if (url.includes('/api/tools/model-explorer/source_robot')) {
        return {
          ok: true,
          json: async () => mockSourceResponse,
        };
      }
      if (url.includes('/api/tools/model-explorer/target_robot')) {
        return {
          ok: true,
          json: async () => mockTargetResponse,
        };
      }
      return { ok: false, status: 404 };
    });
  });

  it('renders standard Model Explorer and toggles Frankenstein Mode', async () => {
    render(<ModelExplorerPage />);

    // Check header
    expect(screen.getByText('Model Explorer')).toBeInTheDocument();

    // Find and check Frankenstein Mode toggle
    const toggle = screen.getByLabelText(/Frankenstein Mode/i);
    expect(toggle).toBeInTheDocument();
    expect(toggle).not.toBeChecked();

    // Toggle Frankenstein Mode
    await act(async () => {
      fireEvent.click(toggle);
    });

    expect(toggle).toBeChecked();
    // In Frankenstein Mode, we should see Source Model and Target Model labels/selectors
    expect(screen.getAllByText('Source Model')[0]).toBeInTheDocument();
    expect(screen.getAllByText('Target Model')[0]).toBeInTheDocument();
  });

  it('loads and displays dual-tree layout in Frankenstein Mode', async () => {
    render(<ModelExplorerPage />);

    // Enable Frankenstein Mode
    const toggle = screen.getByLabelText(/Frankenstein Mode/i);
    await act(async () => {
      fireEvent.click(toggle);
    });

    // Select Source Model
    const sourceSelect = screen.getByLabelText(/Select Source/i);
    await act(async () => {
      fireEvent.change(sourceSelect, { target: { value: 'source_robot' } });
    });

    // Select Target Model
    const targetSelect = screen.getByLabelText(/Select Target/i);
    await act(async () => {
      fireEvent.change(targetSelect, { target: { value: 'target_robot' } });
    });

    // Both tree node lists should render their roots
    expect(screen.getByText('src_root')).toBeInTheDocument();
    expect(screen.getByText('tgt_root')).toBeInTheDocument();
  });

  it('allows copying component from source to target and shows compare diff modal', async () => {
    render(<ModelExplorerPage />);

    const toggle = screen.getByLabelText(/Frankenstein Mode/i);
    await act(async () => {
      fireEvent.click(toggle);
    });

    await act(async () => {
      fireEvent.change(screen.getByLabelText(/Select Source/i), { target: { value: 'source_robot' } });
      fireEvent.change(screen.getByLabelText(/Select Target/i), { target: { value: 'target_robot' } });
    });

    // Click on source node to select it
    await act(async () => {
      fireEvent.click(screen.getByText('src_joint_1'));
    });

    // Click on target node to select it (so copying has a parent target)
    await act(async () => {
      fireEvent.click(screen.getByText('tgt_root'));
    });

    // Copy Component should be visible since src_joint_1 is selected
    const copyBtn = screen.getByLabelText(/Copy Component/i);
    expect(copyBtn).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(copyBtn);
    });

    // Now tgt_root tree should contain the copied joint
    expect(screen.getAllByText(/src_joint_1_copy/i)).toHaveLength(1);

    // Let's open Compare Modal
    const compareBtn = screen.getByRole('button', { name: /Compare/i });
    expect(compareBtn).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(compareBtn);
    });

    // Check that TreeDiffModal is rendered and lists added components in comparison
    expect(screen.getByText(/Model Comparison/i)).toBeInTheDocument();
  });
});

describe('JointManipulator component', () => {
  const mockJoints = [
    {
      id: 'joint_1',
      name: 'shoulder_pitch',
      node_type: 'joint' as const,
      parent_id: 'base',
      children: [],
      properties: { joint_type: 'revolute', lower: -1.5, upper: 1.5 },
    },
    {
      id: 'joint_2',
      name: 'elbow_pitch',
      node_type: 'joint' as const,
      parent_id: 'shoulder',
      children: [],
      properties: { joint_type: 'revolute', lower: -2.0, upper: 2.0 },
    },
  ];

  it('renders joints sliders, reset all button and random pose button', () => {
    const onJointChange = vi.fn();
    const onResetAll = vi.fn();
    const onRandomPose = vi.fn();

    render(
      <JointManipulator
        joints={mockJoints}
        jointValues={{ shoulder_pitch: 0.5, elbow_pitch: -0.2 }}
        onJointChange={onJointChange}
        onResetAll={onResetAll}
        onRandomPose={onRandomPose}
      />
    );

    expect(screen.getByText('shoulder_pitch')).toBeInTheDocument();
    expect(screen.getByText('elbow_pitch')).toBeInTheDocument();

    const resetBtn = screen.getByRole('button', { name: /Reset All/i });
    const randomBtn = screen.getByRole('button', { name: /Random Pose/i });

    expect(resetBtn).toBeInTheDocument();
    expect(randomBtn).toBeInTheDocument();

    fireEvent.click(resetBtn);
    expect(onResetAll).toHaveBeenCalled();

    fireEvent.click(randomBtn);
    expect(onRandomPose).toHaveBeenCalled();
  });
});

