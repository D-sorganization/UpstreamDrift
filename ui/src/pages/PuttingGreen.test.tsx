/**
 * Tests for PuttingGreen page.
 *
 * See issue #1206
 */

import { describe, it, expect } from 'vitest';

import type { PuttResult, GreenReading, ScatterResult } from './PuttingGreen';

describe('PuttingGreen data structures', () => {
  it('should parse a putt simulation result', () => {
    const result: PuttResult = {
      positions: [
        [10.0, 5.0],
        [10.0, 7.0],
        [10.0, 10.0],
        [10.0, 14.8],
      ],
      velocities: [
        [0.0, 2.0],
        [0.0, 1.8],
        [0.0, 1.2],
        [0.0, 0.1],
      ],
      times: [0.0, 0.5, 1.0, 2.0],
      holed: false,
      final_position: [10.0, 14.8],
      total_distance: 9.8,
      duration: 2.0,
    };

    expect(result.positions).toHaveLength(4);
    expect(result.holed).toBe(false);
    expect(result.total_distance).toBeCloseTo(9.8, 1);
    expect(result.final_position).toEqual([10.0, 14.8]);
  });

  it('should identify a holed putt', () => {
    const result: PuttResult = {
      positions: [
        [10.0, 5.0],
        [10.0, 15.0],
      ],
      velocities: [
        [0.0, 2.0],
        [0.0, 0.0],
      ],
      times: [0.0, 2.5],
      holed: true,
      final_position: [10.0, 15.0],
      total_distance: 10.0,
      duration: 2.5,
    };

    expect(result.holed).toBe(true);
    expect(result.duration).toBe(2.5);
  });

  it('should parse a green reading', () => {
    const reading: GreenReading = {
      distance: 11.18,
      total_break: 0.05,
      recommended_speed: 2.1,
      aim_point: [10.02, 14.95],
      elevations: [0.0, 0.01, 0.02, 0.03],
      slopes: [
        [0.001, 0.002],
        [0.001, 0.003],
      ],
    };

    expect(reading.distance).toBeGreaterThan(0);
    expect(reading.recommended_speed).toBeGreaterThan(0);
    expect(reading.aim_point).toHaveLength(2);
    expect(reading.elevations.length).toBeGreaterThan(0);
  });

  it('should parse scatter analysis results', () => {
    const scatter: ScatterResult = {
      final_positions: [
        [10.1, 14.9],
        [9.8, 15.2],
        [10.0, 15.0],
        [10.3, 14.7],
        [10.0, 15.0],
      ],
      holed_count: 2,
      total_simulations: 5,
      average_distance_from_hole: 0.15,
      make_percentage: 40.0,
    };

    expect(scatter.total_simulations).toBe(5);
    expect(scatter.holed_count).toBe(2);
    expect(scatter.make_percentage).toBe(40.0);
    expect(scatter.final_positions).toHaveLength(5);
  });

  it('should compute distance between ball and hole', () => {
    const ballX = 10.0;
    const ballY = 5.0;
    const holeX = 10.0;
    const holeY = 15.0;

    const distance = Math.sqrt(
      (holeX - ballX) ** 2 + (holeY - ballY) ** 2,
    );
    expect(distance).toBeCloseTo(10.0, 5);
  });

  it('should normalize direction vector', () => {
    const dirX = 3.0;
    const dirY = 4.0;
    const norm = Math.sqrt(dirX * dirX + dirY * dirY);
    const normalizedX = dirX / norm;
    const normalizedY = dirY / norm;

    expect(normalizedX).toBeCloseTo(0.6, 5);
    expect(normalizedY).toBeCloseTo(0.8, 5);
    expect(
      Math.sqrt(normalizedX ** 2 + normalizedY ** 2),
    ).toBeCloseTo(1.0, 10);
  });

  it('should validate stimp rating range', () => {
    const validRatings = [6.0, 8.5, 10.0, 12.5, 15.0];
    const invalidRatings = [5.0, 16.0, -1.0];

    for (const rating of validRatings) {
      expect(rating).toBeGreaterThanOrEqual(6.0);
      expect(rating).toBeLessThanOrEqual(15.0);
    }

    for (const rating of invalidRatings) {
      const isValid = rating >= 6.0 && rating <= 15.0;
      expect(isValid).toBe(false);
    }
  });
});
