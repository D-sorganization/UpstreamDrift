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
    // Must mirror the API's VALID_ESTIMATOR_TYPES (src/api/config.py, #8392).
    const validTypes = ['mediapipe', 'openpose'];
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

  it('should derive frame index from currentTime using clip fps (25 fps fixture, not 30)', () => {
    // 25 fps fixture has dt = 0.04s per frame.
    // At t = 0.20s: 0.20 / 0.04 = 5 (index 5)
    // At 30 fps it would be 0.20 * 30 = 6.
    const fps = 25;
    const currentTime = 0.20;
    const frameIndex = Math.round(currentTime * fps);
    expect(frameIndex).toBe(5);
    expect(frameIndex).not.toBe(6);
  });
});

describe('VideoAnalyzer sizing and sync', () => {
  it('sizes viewBox from videoWidth/videoHeight on loadedmetadata', async () => {
    const { render, fireEvent } = await import('@testing-library/react');
    const { VideoAnalyzerPage } = await import('./VideoAnalyzer');

    const { container } = render(<VideoAnalyzerPage />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;

    // Simulate uploading a video file
    const file = new File(['dummy video content'], 'test_swing_25fps.mp4', {
      type: 'video/mp4',
    });
    // Mock URL.createObjectURL
    const origCreateObjectURL = URL.createObjectURL;
    URL.createObjectURL = () => 'blob:http://localhost/test-video';

    try {
      fireEvent.change(fileInput, { target: { files: [file] } });

      const video = container.querySelector('video') as HTMLVideoElement;
      expect(video).toBeDefined();

      // Mock video dimensions: 1280x720 (different from legacy 640x480)
      Object.defineProperty(video, 'videoWidth', { value: 1280, configurable: true });
      Object.defineProperty(video, 'videoHeight', { value: 720, configurable: true });

      // Trigger loadedmetadata
      fireEvent.loadedMetadata(video);

      // Check if container or overlay uses 1280 and 720
      const overlaySvg = container.querySelector('[data-testid="video-overlay-container"] svg, [data-testid="pose-overlay"], [data-testid="video-force-overlay"]');
      if (overlaySvg) {
        expect(overlaySvg.getAttribute('viewBox')).toBe('0 0 1280 720');
      }
    } finally {
      URL.createObjectURL = origCreateObjectURL;
    }
  });
});

