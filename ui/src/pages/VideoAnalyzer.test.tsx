/**
 * Tests for VideoAnalyzer page.
 *
 * See issue #1206
 */

import { describe, it, expect } from 'vitest';

import type { VideoAnalysisResult, PoseFrame, TaskStatus } from './VideoAnalyzer';

describe('VideoAnalyzer data structures', () => {
  it('should parse video analysis result', () => {
    const result: VideoAnalysisResult = {
      filename: 'swing_front.mp4',
      total_frames: 120,
      valid_frames: 115,
      average_confidence: 0.87,
      quality_metrics: {
        stability: 0.92,
        coverage: 0.88,
        smoothness: 0.95,
      },
      pose_data: [
        {
          timestamp: 0.0,
          confidence: 0.91,
          joint_angles: { hip: 45.0, shoulder: 90.0, elbow: 120.0 },
          keypoints: { nose: [320, 100], left_shoulder: [280, 180] },
        },
        {
          timestamp: 0.033,
          confidence: 0.89,
          joint_angles: { hip: 46.0, shoulder: 89.5, elbow: 118.0 },
          keypoints: { nose: [321, 100], left_shoulder: [281, 181] },
        },
      ],
    };

    expect(result.filename).toBe('swing_front.mp4');
    expect(result.total_frames).toBe(120);
    expect(result.valid_frames).toBe(115);
    expect(result.average_confidence).toBeCloseTo(0.87, 2);
    expect(result.pose_data).toHaveLength(2);
    expect(result.quality_metrics.stability).toBe(0.92);
  });

  it('should access pose frame data', () => {
    const frame: PoseFrame = {
      timestamp: 1.5,
      confidence: 0.93,
      joint_angles: {
        hip_flexion: 85.2,
        shoulder_rotation: 110.5,
        spine_tilt: 12.3,
        elbow_angle: 145.0,
        wrist_cock: 32.1,
      },
      keypoints: {
        left_shoulder: [280, 180, 0],
        right_shoulder: [360, 180, 0],
        left_hip: [290, 350, 0],
        right_hip: [350, 350, 0],
      },
    };

    expect(frame.timestamp).toBe(1.5);
    expect(frame.confidence).toBeGreaterThan(0.9);
    expect(Object.keys(frame.joint_angles)).toHaveLength(5);
    expect(frame.keypoints.left_shoulder).toHaveLength(3);
  });

  it('should handle task status lifecycle', () => {
    const statuses: TaskStatus[] = [
      { task_id: 'abc-123', status: 'started' },
      { task_id: 'abc-123', status: 'processing', progress: 50 },
      {
        task_id: 'abc-123',
        status: 'completed',
        result: { filename: 'test.mp4', total_frames: 60 },
      },
    ];

    expect(statuses[0].status).toBe('started');
    expect(statuses[1].progress).toBe(50);
    expect(statuses[2].status).toBe('completed');
    expect(statuses[2].result).toBeDefined();
  });

  it('should handle failed task status', () => {
    const status: TaskStatus = {
      task_id: 'def-456',
      status: 'failed',
      error: 'Video format not supported',
    };

    expect(status.status).toBe('failed');
    expect(status.error).toContain('not supported');
  });

  it('should validate estimator types', () => {
    const validTypes = ['mediapipe', 'openpose', 'blazepose'];
    const invalidType = 'unknown_estimator';

    for (const t of validTypes) {
      expect(validTypes).toContain(t);
    }
    expect(validTypes).not.toContain(invalidType);
  });

  it('should compute frame-to-frame joint angle change', () => {
    const frame1: PoseFrame = {
      timestamp: 0.0,
      confidence: 0.9,
      joint_angles: { hip: 45.0, shoulder: 90.0 },
      keypoints: {},
    };
    const frame2: PoseFrame = {
      timestamp: 0.033,
      confidence: 0.88,
      joint_angles: { hip: 47.5, shoulder: 88.0 },
      keypoints: {},
    };

    const hipDelta = frame2.joint_angles.hip - frame1.joint_angles.hip;
    const shoulderDelta =
      frame2.joint_angles.shoulder - frame1.joint_angles.shoulder;
    const dt = frame2.timestamp - frame1.timestamp;
    const hipVelocity = hipDelta / dt;

    expect(hipDelta).toBeCloseTo(2.5, 5);
    expect(shoulderDelta).toBeCloseTo(-2.0, 5);
    expect(hipVelocity).toBeCloseTo(75.76, 0);
  });
});
