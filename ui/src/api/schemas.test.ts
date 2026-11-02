import { describe, expect, it } from 'vitest';
import {
  parseActiveTerrain, parseTerrainLoadResponse, parseTerrainMaterials,
  parseTerrainPresets, parseTerrainTypes, ResponseValidationError,
} from './schemas';

describe('terrain response schemas', () => {
  it('normalizes backend terrain and catalog payloads', () => {
    const active = parseActiveTerrain({
      name: 'default_fairway', width_m: 100, length_m: 200,
      resolution_m: 1, default_type: 'rough', patch_count: 1, region_count: 0,
    });
    const [preset] = parseTerrainPresets([{
      name: 'putting_green', description: 'Putting green',
      terrain_types: ['green', 'fringe'], width_m: 10, length_m: 15,
    }]);
    const [material] = parseTerrainMaterials([{
      name: 'green', friction_coefficient: 0.35, restitution: 0.7,
      rolling_resistance: 0.05, hardness: 0.8, grass_height_m: 0.004,
      compressibility: 0.05,
    }]);

    expect(active).toMatchObject({ id: 'default_fairway', terrain_type: 'rough', properties: {} });
    expect(active.properties.dimensions).toBeUndefined();
    expect(preset).toMatchObject({ id: 'putting_green', terrain_type: 'green, fringe', defaults: { width_m: 10, length_m: 15 } });
    expect(material).toMatchObject({ id: 'green', friction_coefficient: 0.35 });
    expect(parseTerrainTypes(['green', 'fairway']).map(({ id }) => id)).toEqual(['green', 'fairway']);
  });

  it('rejects invalid active, load, and collection payloads', () => {
    expect(() => parseActiveTerrain({ name: 42 })).toThrow(ResponseValidationError);
    expect(() => parseTerrainLoadResponse({ success: false, error: 'Unknown preset' })).toThrow('Unknown preset');
    expect(() => parseTerrainTypes(['green', 3])).toThrow(ResponseValidationError);
  });
});
