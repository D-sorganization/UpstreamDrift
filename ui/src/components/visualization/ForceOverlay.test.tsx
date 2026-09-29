/**
 * Tests for ForceOverlay component.
 *
 * See issue #1199
 */

import { describe, it, expect } from 'vitest';
import type { ForceVector3D, ForceOverlayConfig } from './ForceOverlay';

describe('ForceOverlay types', () => {
  it('should define ForceVector3D interface correctly', () => {
    const vector: ForceVector3D = {
      body_name: 'torso',
      force_type: 'applied',
      origin: [0, 1, 0],
      direction: [1, 0, 0],
      magnitude: 50.0,
      color: [1, 0, 0, 1],
      label: '50.0 N*m',
    };

    expect(vector.body_name).toBe('torso');
    expect(vector.force_type).toBe('applied');
    expect(vector.magnitude).toBe(50.0);
    expect(vector.origin).toEqual([0, 1, 0]);
    expect(vector.direction).toEqual([1, 0, 0]);
    expect(vector.color).toEqual([1, 0, 0, 1]);
    expect(vector.label).toBe('50.0 N*m');
  });

  it('should allow null label on ForceVector3D', () => {
    const vector: ForceVector3D = {
      body_name: 'arm',
      force_type: 'gravity',
      origin: [0, 0.5, 0],
      direction: [0, -1, 0],
      magnitude: 9.81,
      color: [0, 0, 1, 1],
      label: null,
    };

    expect(vector.label).toBeNull();
  });

  it('should define ForceOverlayConfig with defaults', () => {
    const config: ForceOverlayConfig = {
      enabled: true,
      forceTypes: ['applied', 'gravity'],
      scaleFactor: 0.01,
      colorByMagnitude: true,
      showLabels: false,
      bodyFilter: null,
    };

    expect(config.enabled).toBe(true);
    expect(config.forceTypes).toHaveLength(2);
    expect(config.scaleFactor).toBe(0.01);
    expect(config.bodyFilter).toBeNull();
  });

  it('should support body filtering', () => {
    const config: ForceOverlayConfig = {
      enabled: true,
      forceTypes: ['applied'],
      scaleFactor: 0.01,
      colorByMagnitude: false,
      showLabels: true,
      bodyFilter: ['torso', 'hand'],
    };

    expect(config.bodyFilter).toEqual(['torso', 'hand']);
  });

  it('should support all force types', () => {
    const types: ForceVector3D['force_type'][] = [
      'applied',
      'gravity',
      'contact',
      'bias',
    ];

    for (const type of types) {
      const vec: ForceVector3D = {
        body_name: 'test',
        force_type: type,
        origin: [0, 0, 0],
        direction: [0, 1, 0],
        magnitude: 1.0,
        color: [1, 1, 1, 1],
      };
      expect(vec.force_type).toBe(type);
    }
  });
});
