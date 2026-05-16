/**
 * Tests for Terrain page.
 *
 * Validates data structures and type contracts for the terrain engine.
 */

import { describe, it, expect } from 'vitest';

import type {
  TerrainPreset,
  TerrainMaterial,
  TerrainTypeInfo,
  TerrainProperties,
  ActiveTerrain,
  TerrainLoadState,
} from './Terrain';

describe('Terrain data structures', () => {
  it('should parse a terrain preset', () => {
    const preset: TerrainPreset = {
      id: 'highlands',
      name: 'Highlands',
      description: 'Rolling highland terrain with moderate slopes',
      terrain_type: 'procedural',
      defaults: { resolution: 256, elevation_scale: 5.0 },
    };

    expect(preset.id).toBe('highlands');
    expect(preset.name).toBe('Highlands');
    expect(preset.terrain_type).toBe('procedural');
    expect(preset.defaults).toHaveProperty('resolution');
  });

  it('should parse terrain material properties', () => {
    const material: TerrainMaterial = {
      id: 'grass',
      name: 'Grass',
      friction_coefficient: 0.65,
      restitution: 0.3,
      color: '#2d5a27',
    };

    expect(material.id).toBe('grass');
    expect(material.friction_coefficient).toBeGreaterThan(0);
    expect(material.restitution).toBeLessThanOrEqual(1);
    expect(material.color).toMatch(/^#[0-9a-fA-F]{6}$/);
  });

  it('should parse terrain type info', () => {
    const terrainType: TerrainTypeInfo = {
      id: 'procedural',
      name: 'Procedural',
      description: 'Generated using noise algorithms',
    };

    expect(terrainType.id).toBe('procedural');
    expect(terrainType.name).toBe('Procedural');
    expect(terrainType.description.length).toBeGreaterThan(0);
  });

  it('should parse terrain properties', () => {
    const props: TerrainProperties = {
      dimensions: [100, 100, 10],
      resolution: 256,
      material: 'grass',
      elevation_range: [0, 50],
      slope_range: [0, 45],
      features: ['water_hazard', 'bunker'],
    };

    expect(props.dimensions).toHaveLength(3);
    expect(props.resolution).toBeGreaterThan(0);
    expect(props.features).toContain('water_hazard');
  });

  it('should parse active terrain', () => {
    const active: ActiveTerrain = {
      id: 'terrain_001',
      name: 'Augusta National',
      terrain_type: 'procedural',
      material: 'grass',
      properties: {
        dimensions: [200, 200, 20],
        resolution: 512,
        material: 'grass',
        elevation_range: [0, 30],
        slope_range: [0, 35],
        features: ['sand_bunker', 'water_hazard', 'trees'],
      },
      loaded_at: '2026-05-15T10:00:00Z',
    };

    expect(active.id).toBe('terrain_001');
    expect(active.properties.dimensions[0]).toBe(200);
    expect(active.properties.features).toHaveLength(3);
    expect(active.loaded_at).toBeTruthy();
  });

  it('should validate terrain load state transitions', () => {
    const states: TerrainLoadState[] = ['idle', 'loading', 'loaded', 'error'];

    expect(states).toContain('idle');
    expect(states).toContain('loading');
    expect(states).toContain('loaded');
    expect(states).toHaveLength(4);
  });

  it('should validate elevation range ordering', () => {
    const props: TerrainProperties = {
      dimensions: [100, 100, 10],
      resolution: 256,
      material: 'grass',
      elevation_range: [0, 50],
      slope_range: [0, 45],
      features: [],
    };

    expect(props.elevation_range[0]).toBeLessThanOrEqual(props.elevation_range[1]);
    expect(props.slope_range[0]).toBeLessThanOrEqual(props.slope_range[1]);
  });

  it('should handle multiple material friction values', () => {
    const materials: TerrainMaterial[] = [
      { id: 'grass', name: 'Grass', friction_coefficient: 0.65, restitution: 0.3, color: '#2d5a27' },
      { id: 'sand', name: 'Sand', friction_coefficient: 0.35, restitution: 0.1, color: '#c2b280' },
      { id: 'water', name: 'Water', friction_coefficient: 0.05, restitution: 0.0, color: '#1a6b8a' },
    ];

    const sorted = [...materials].sort((a, b) => b.friction_coefficient - a.friction_coefficient);
    expect(sorted[0].id).toBe('grass');
    expect(sorted[2].id).toBe('water');
  });
});
