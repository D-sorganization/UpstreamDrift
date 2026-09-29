/**
 * Tests for MotionCapture page.
 *
 * See issues #1206, #7454, #8406
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

vi.mock('@/api/fetch', () => ({
  apiFetch: vi.fn(),
  apiFetchForm: vi.fn(),
}));

// #8406: the live pose channel is mocked so no real WebSocket is opened.
vi.mock('@/hooks/useRealtimeChannel', () => ({
  useRealtimeChannel: vi.fn(() => ({ message: null, status: 'connecting' })),
}));

// #8406: the 3D skeleton is lazy-loaded (keeps three.js out of the route
// chunk); mock the module so the toggle can be tested without WebGL.
vi.mock('@/components/visualization/MocapSkeleton3D', () => ({
  default: ({ joints }: { joints: unknown[] }) => (
    <div data-testid="mocap-skeleton-3d-mock" data-joint-count={joints.length} />
  ),
}));

import { apiFetch } from '@/api/fetch';
import { useRealtimeChannel } from '@/hooks/useRealtimeChannel';
import { extractSkeletonJoints } from '@/utils/skeletonJoints';
import { MotionCapturePage } from './MotionCapture';
import type {
  CaptureSource,
  JointData,
  RecordingInfo,
  CaptureSession,
  PlaybackState,
} from './MotionCapture';

const mockedApiFetch = vi.mocked(apiFetch);
const mockedUseRealtimeChannel = vi.mocked(useRealtimeChannel);

function mockApi(sources: CaptureSource[], skeleton: JointData[] = []) {
  mockedApiFetch.mockImplementation(async (path: string) => {
    if (path.endsWith('/sources')) return sources as unknown;
    if (path.includes('/skeleton/')) return skeleton as unknown;
    if (path.endsWith('/recordings')) return [] as unknown;
    throw new Error(`unexpected path ${path}`);
  });
}

const API_SOURCES: CaptureSource[] = [
  {
    id: 'mediapipe',
    name: 'MediaPipe Pose',
    type: 'mediapipe',
    available: false,
    reason: 'MediaPipe is not installed on the server (pip install mediapipe)',
    description: 'Real-time pose estimation using Google MediaPipe',
  },
  {
    id: 'openpose',
    name: 'OpenPose',
    type: 'openpose',
    available: false,
    reason: 'OpenPose Python bindings are not installed on the server',
    description: 'Multi-person pose estimation using OpenPose',
  },
  {
    id: 'c3d',
    name: 'C3D File Import',
    type: 'c3d',
    available: true,
    reason: null,
    description: 'Import motion capture data from C3D files',
  },
];

describe('MotionCapturePage source-driven UI (#7454)', () => {
  beforeEach(() => {
    mockedApiFetch.mockReset();
  });

  it('renders all sources from the API, disabling unavailable ones with reason', async () => {
    mockApi(API_SOURCES);
    render(<MotionCapturePage />);

    const mediapipe = await screen.findByTestId('source-mediapipe');
    expect(mediapipe).toBeDisabled();
    expect(screen.getByTestId('source-reason-mediapipe')).toHaveTextContent(
      'MediaPipe is not installed',
    );

    const openpose = screen.getByTestId('source-openpose');
    expect(openpose).toBeDisabled();
    expect(screen.getByTestId('source-reason-openpose')).toHaveTextContent(
      'OpenPose Python bindings are not installed',
    );

    expect(screen.getByTestId('source-c3d')).toBeEnabled();
    expect(screen.queryByTestId('source-reason-c3d')).toBeNull();
  });

  it('auto-selects the first available source and shows the C3D upload control', async () => {
    mockApi(API_SOURCES);
    render(<MotionCapturePage />);

    // c3d is the only available source -> auto-selected -> upload input shown
    expect(await screen.findByTestId('c3d-file-input')).toBeInTheDocument();
  });

  it('does not show the C3D upload control for non-c3d sources', async () => {
    mockApi(
      API_SOURCES.map((s) =>
        s.id === 'mediapipe' ? { ...s, available: true, reason: null } : s,
      ),
    );
    render(<MotionCapturePage />);

    await screen.findByTestId('source-mediapipe');
    expect(screen.queryByTestId('c3d-file-input')).toBeNull();
  });

  it('shows a loading placeholder when no sources have arrived', () => {
    mockedApiFetch.mockImplementation(() => new Promise(() => {}));
    render(<MotionCapturePage />);
    expect(screen.getByText('Loading sources...')).toBeInTheDocument();
  });
});

const FLAT_JOINTS: JointData[] = [
  { name: 'hips', position: [0, 0.5, 0], confidence: 0.9, parent: null },
  { name: 'head', position: [0, 0.9, 0], confidence: 0.8, parent: 'hips' },
];

const DEPTH_JOINTS: JointData[] = [
  { name: 'hips', position: [0, 0.5, 0.3], confidence: 0.9, parent: null },
  { name: 'head', position: [0, 0.9, 0.35], confidence: 0.8, parent: 'hips' },
];

describe('MotionCapturePage 2D/3D skeleton views (#8406)', () => {
  beforeEach(() => {
    mockedApiFetch.mockReset();
    mockedUseRealtimeChannel.mockReset();
    mockedUseRealtimeChannel.mockReturnValue({
      message: null,
      status: 'connecting',
    });
  });

  it('defaults to the 2D SVG view when frame data has no z values', async () => {
    mockApi(API_SOURCES, FLAT_JOINTS);
    render(<MotionCapturePage />);

    // Wait for the skeleton template to load into the page
    await screen.findByText('2 joints detected');

    expect(screen.getByTestId('skeleton-renderer')).toBeInTheDocument();
    expect(screen.queryByTestId('mocap-skeleton-3d-mock')).toBeNull();
  });

  it('defaults to the 3D view when frame data carries z values', async () => {
    mockApi(API_SOURCES, DEPTH_JOINTS);
    render(<MotionCapturePage />);

    const view3d = await screen.findByTestId('mocap-skeleton-3d-mock');
    expect(view3d).toHaveAttribute('data-joint-count', '2');
    expect(screen.queryByTestId('skeleton-renderer')).toBeNull();
  });

  it('toggles between 2D and 3D views', async () => {
    mockApi(API_SOURCES, DEPTH_JOINTS);
    render(<MotionCapturePage />);

    await screen.findByTestId('mocap-skeleton-3d-mock');

    fireEvent.click(screen.getByTestId('view-toggle-2d'));
    expect(screen.getByTestId('skeleton-renderer')).toBeInTheDocument();
    expect(screen.queryByTestId('mocap-skeleton-3d-mock')).toBeNull();

    fireEvent.click(screen.getByTestId('view-toggle-3d'));
    await waitFor(() => {
      expect(screen.getByTestId('mocap-skeleton-3d-mock')).toBeInTheDocument();
    });
    expect(screen.queryByTestId('skeleton-renderer')).toBeNull();
  });

  it('a manual 2D override wins over depth-based auto-selection', async () => {
    mockApi(API_SOURCES, FLAT_JOINTS);
    render(<MotionCapturePage />);
    await screen.findByText('2 joints detected');

    fireEvent.click(screen.getByTestId('view-toggle-3d'));
    await waitFor(() => {
      expect(screen.getByTestId('mocap-skeleton-3d-mock')).toBeInTheDocument();
    });
  });
});

describe('MotionCapturePage live pose streaming (#8406)', () => {
  beforeEach(() => {
    mockedApiFetch.mockReset();
    mockedUseRealtimeChannel.mockReset();
  });

  it('subscribes to the pose/canonical channel', async () => {
    mockedUseRealtimeChannel.mockReturnValue({
      message: null,
      status: 'connecting',
    });
    mockApi(API_SOURCES);
    render(<MotionCapturePage />);
    await screen.findByTestId('source-c3d');

    expect(mockedUseRealtimeChannel).toHaveBeenCalledWith('pose/canonical');
  });

  it('live joint messages drive the skeleton view instead of polled data', async () => {
    mockedUseRealtimeChannel.mockReturnValue({
      message: {
        joints: [
          {
            name: 'live_hip',
            position: [0, 0.5, 0.2],
            confidence: 0.7,
            parent: null,
          },
        ],
      },
      status: 'connected',
    });
    mockApi(API_SOURCES, FLAT_JOINTS);
    render(<MotionCapturePage />);

    // Live frame carries depth -> 3D view with the single live joint
    const view3d = await screen.findByTestId('mocap-skeleton-3d-mock');
    expect(view3d).toHaveAttribute('data-joint-count', '1');
    expect(screen.getByTestId('live-pose-indicator')).toBeInTheDocument();
    // Right panel lists the live joint
    expect(screen.getByText('live_hip')).toBeInTheDocument();
  });

  it('falls back to polled joints when the channel payload has no joints', async () => {
    mockedUseRealtimeChannel.mockReturnValue({
      // Pose Studio publishes joint angles, not positions — not renderable
      message: { joint_angles_deg: { hip: 12 } },
      status: 'connected',
    });
    mockApi(API_SOURCES, FLAT_JOINTS);
    render(<MotionCapturePage />);

    await screen.findByText('2 joints detected');
    expect(screen.queryByTestId('live-pose-indicator')).toBeNull();
    expect(screen.getByTestId('skeleton-renderer')).toBeInTheDocument();
  });
});

describe('extractSkeletonJoints (#8406)', () => {
  it('accepts a SkeletonFrame-shaped payload', () => {
    const joints = extractSkeletonJoints({
      frame_index: 3,
      joints: [
        { name: 'hips', position: [0, 1, 0.2], confidence: 0.9, parent: null },
        { name: 'head', position: [0, 1.6, 0.2], parent: 'hips' },
      ],
    });
    expect(joints).toHaveLength(2);
    expect(joints?.[1]).toEqual({
      name: 'head',
      position: [0, 1.6, 0.2],
      confidence: 1.0,
      parent: 'hips',
    });
  });

  it('accepts a bare joint array', () => {
    const joints = extractSkeletonJoints([
      { name: 'hips', position: [0, 1, 0] },
    ]);
    expect(joints).toHaveLength(1);
  });

  it('rejects payloads without positional joints', () => {
    expect(extractSkeletonJoints(null)).toBeNull();
    expect(extractSkeletonJoints(undefined)).toBeNull();
    expect(extractSkeletonJoints({ joint_angles_deg: { hip: 5 } })).toBeNull();
    expect(extractSkeletonJoints({ joints: [] })).toBeNull();
    expect(extractSkeletonJoints({ joints: [{ name: 'x' }] })).toBeNull();
    expect(
      extractSkeletonJoints({ joints: [{ name: 'x', position: ['a'] }] }),
    ).toBeNull();
    expect(extractSkeletonJoints('nope')).toBeNull();
  });
});

describe('MotionCapture data structures', () => {
  it('should parse capture sources', () => {
    const sources: CaptureSource[] = [
      {
        id: 'mediapipe',
        name: 'MediaPipe Pose',
        type: 'mediapipe',
        available: true,
        description: 'Real-time pose estimation using Google MediaPipe',
      },
      {
        id: 'openpose',
        name: 'OpenPose',
        type: 'openpose',
        available: false,
        description: 'Multi-person pose estimation using OpenPose',
      },
      {
        id: 'c3d',
        name: 'C3D File Import',
        type: 'c3d',
        available: true,
        description: 'Import motion capture data from C3D files',
      },
    ];

    expect(sources).toHaveLength(3);
    expect(sources[0].available).toBe(true);
    expect(sources[1].available).toBe(false);
    expect(sources[2].type).toBe('c3d');
  });

  it('should parse joint data with parent hierarchy', () => {
    const joints: JointData[] = [
      { name: 'nose', position: [0.0, 0.8, 0.0], confidence: 0.95, parent: null },
      {
        name: 'left_shoulder',
        position: [-0.15, 0.6, 0.0],
        confidence: 0.92,
        parent: 'nose',
      },
      {
        name: 'left_elbow',
        position: [-0.25, 0.4, 0.0],
        confidence: 0.88,
        parent: 'left_shoulder',
      },
      {
        name: 'left_wrist',
        position: [-0.3, 0.2, 0.0],
        confidence: 0.85,
        parent: 'left_elbow',
      },
    ];

    expect(joints).toHaveLength(4);
    expect(joints[0].parent).toBeNull(); // root
    expect(joints[1].parent).toBe('nose');
    expect(joints[3].confidence).toBeCloseTo(0.85, 2);
  });

  it('should build joint hierarchy map', () => {
    const joints: JointData[] = [
      { name: 'nose', position: [0, 0, 0], confidence: 1.0, parent: null },
      { name: 'left_shoulder', position: [0, 0, 0], confidence: 1.0, parent: 'nose' },
      { name: 'right_shoulder', position: [0, 0, 0], confidence: 1.0, parent: 'nose' },
      {
        name: 'left_elbow',
        position: [0, 0, 0],
        confidence: 1.0,
        parent: 'left_shoulder',
      },
    ];

    const map = new Map(joints.map((j) => [j.name, j]));
    expect(map.size).toBe(4);
    expect(map.get('nose')?.parent).toBeNull();
    expect(map.get('left_elbow')?.parent).toBe('left_shoulder');

    // Trace chain: left_elbow -> left_shoulder -> nose
    const chain: string[] = [];
    let current = map.get('left_elbow');
    while (current) {
      chain.push(current.name);
      current = current.parent ? map.get(current.parent) : undefined;
    }
    expect(chain).toEqual(['left_elbow', 'left_shoulder', 'nose']);
  });

  it('should parse recording info', () => {
    const recording: RecordingInfo = {
      name: 'recording_session_1',
      source_type: 'mediapipe',
      total_frames: 300,
      duration_seconds: 10.0,
      frame_rate: 30.0,
      joint_names: [
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
      ],
    };

    expect(recording.total_frames).toBe(300);
    expect(recording.duration_seconds).toBe(10.0);
    expect(recording.frame_rate).toBe(30.0);
    expect(recording.joint_names).toContain('left_shoulder');
  });

  it('should handle capture session lifecycle', () => {
    const started: CaptureSession = {
      session_id: 'session_1',
      status: 'recording',
      source_type: 'mediapipe',
      message: 'Capture session started with mediapipe at 30 fps',
    };

    const stopped: CaptureSession = {
      session_id: 'session_1',
      status: 'stopped',
      source_type: 'mediapipe',
      message: "Session stopped. Recording saved as 'recording_session_1'",
    };

    expect(started.status).toBe('recording');
    expect(stopped.status).toBe('stopped');
    expect(stopped.message).toContain('saved');
  });

  it('should track playback state', () => {
    const states: PlaybackState[] = [
      {
        recording_name: 'rec_1',
        status: 'playing',
        current_frame: 0,
        total_frames: 300,
      },
      {
        recording_name: 'rec_1',
        status: 'paused',
        current_frame: 150,
        total_frames: 300,
      },
      {
        recording_name: 'rec_1',
        status: 'stopped',
        current_frame: 0,
        total_frames: 300,
      },
    ];

    expect(states[0].status).toBe('playing');
    expect(states[1].current_frame).toBe(150);
    expect(states[2].status).toBe('stopped');
  });

  it('should compute distance between two joints', () => {
    const shoulder: JointData = {
      name: 'shoulder',
      position: [0.0, 0.6, 0.0],
      confidence: 0.95,
      parent: null,
    };
    const elbow: JointData = {
      name: 'elbow',
      position: [0.25, 0.4, 0.05],
      confidence: 0.9,
      parent: 'shoulder',
    };

    const dx = elbow.position[0] - shoulder.position[0];
    const dy = elbow.position[1] - shoulder.position[1];
    const dz = elbow.position[2] - shoulder.position[2];
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    expect(distance).toBeGreaterThan(0);
    expect(distance).toBeCloseTo(0.323, 2);
  });

  it('should validate source types', () => {
    const validTypes = new Set(['mediapipe', 'openpose', 'c3d']);

    expect(validTypes.has('mediapipe')).toBe(true);
    expect(validTypes.has('openpose')).toBe(true);
    expect(validTypes.has('c3d')).toBe(true);
    expect(validTypes.has('kinect')).toBe(false);
  });

  it('should filter joints by confidence threshold', () => {
    const joints: JointData[] = [
      { name: 'nose', position: [0, 0, 0], confidence: 0.95, parent: null },
      { name: 'left_hip', position: [0, 0, 0], confidence: 0.3, parent: null },
      { name: 'right_hip', position: [0, 0, 0], confidence: 0.85, parent: null },
      { name: 'left_foot', position: [0, 0, 0], confidence: 0.1, parent: null },
    ];

    const highConfidence = joints.filter((j) => j.confidence > 0.5);
    expect(highConfidence).toHaveLength(2);
    expect(highConfidence.map((j) => j.name)).toContain('nose');
    expect(highConfidence.map((j) => j.name)).toContain('right_hip');
  });
});
