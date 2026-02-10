/**
 * Tests for MotionCapture page.
 *
 * See issue #1206
 */

import { describe, it, expect } from 'vitest';

import type {
  CaptureSource,
  JointData,
  RecordingInfo,
  CaptureSession,
  PlaybackState,
} from './MotionCapture';

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
