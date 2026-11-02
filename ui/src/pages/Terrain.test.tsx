/**
 * Tests for Terrain page.
 *
 * Validates data structures and type contracts for the terrain engine.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { TerrainPage } from './Terrain';

import type {
  TerrainPreset,
  TerrainMaterial,
  TerrainTypeInfo,
  TerrainProperties,
  ActiveTerrain,
  TerrainLoadState,
} from './Terrain';

const backendTerrainResponses: Record<string, unknown> = {
  '/api/terrain/presets': [{ name: 'putting_green', description: 'Putting green', terrain_types: ['green'], width_m: 10, length_m: 15 }],
  '/api/terrain/materials': [{ name: 'green', friction_coefficient: 0.35, restitution: 0.7 }],
  '/api/terrain/types': ['green', 'fairway'],
  '/api/terrain/active': { name: 'default_fairway', default_type: 'rough' },
};

let activeTerrainNotFound = false;
let active404Delay = 0;
let activeTerrainFailure: { status: number; detail: string } | null = null;
let presetCatalogFailure = false;
let terrainLoadResponse: unknown = {
  success: true,
  name: 'loaded_green',
  width_m: 10,
  length_m: 15,
  terrain_types: ['green', 'fringe'],
};
const jsonResponse = (body: unknown, status = 200): Response => ({
  ok: status < 400, status, statusText: status === 404 ? 'Not Found' : 'Test response',
  json: async () => body,
}) as Response;
const mockFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
  const path = String(input);
  if (path === '/api/terrain/active' && activeTerrainNotFound) {
    if (active404Delay) await new Promise((resolve) => setTimeout(resolve, active404Delay));
    return jsonResponse({ detail: 'No active terrain' }, 404);
  }
  if (path === '/api/terrain/active' && activeTerrainFailure) {
    return jsonResponse({ detail: activeTerrainFailure.detail }, activeTerrainFailure.status);
  }
  if (path === '/api/terrain/presets' && presetCatalogFailure) {
    return jsonResponse({ detail: 'Preset catalog unavailable' }, 503);
  }
  return jsonResponse(path === '/api/terrain/load' && init?.method === 'POST' ? terrainLoadResponse : backendTerrainResponses[path] ?? {});
});

beforeEach(() => {
  mockFetch.mockClear();
  vi.stubGlobal('fetch', mockFetch);
});

afterEach(() => {
  activeTerrainNotFound = false;
  active404Delay = 0;
  activeTerrainFailure = null;
  presetCatalogFailure = false;
  terrainLoadResponse = {
    success: true,
    name: 'loaded_green',
    width_m: 10,
    length_m: 15,
    terrain_types: ['green', 'fringe'],
  };
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('TerrainPage backend contract', () => {
  it('renders the backend active terrain without properties and has no list key warnings', async () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    render(<TerrainPage />);

    expect(await screen.findByText('default_fairway')).toBeInTheDocument();
    expect(within(screen.getByText('Dimensions').parentElement!).getByText('N/A'))
      .toBeInTheDocument();
    expect(within(screen.getByText('Resolution').parentElement!).getByText('N/A'))
      .toBeInTheDocument();
    expect(within(screen.getByText('Slope Range').parentElement!).getByText('N/A'))
      .toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'materials' }));
    expect(await screen.findByText('Friction: 0.35')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'types' }));
    expect(within(screen.getByRole('complementary')).getByText('fairway'))
      .toBeInTheDocument();

    expect(consoleError.mock.calls.flat().join(' ')).not.toMatch(
      /unique "key"|Each child in a list/i,
    );
  });

  it('renders backend lists without React key warnings when no terrain is active', async () => {
    activeTerrainNotFound = true;
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    const view = render(<TerrainPage />);

    expect(await screen.findByText('putting_green')).toBeInTheDocument();
    expect(screen.queryByText('No active terrain')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'materials' }));
    expect(await screen.findByText('Friction: 0.35')).toBeInTheDocument();
    expect(consoleError.mock.calls.flat().join(' ')).not.toMatch(
      /unique "key"|Each child in a list/i,
    );
    fireEvent.click(screen.getByRole('button', { name: 'types' }));
    expect(await screen.findByText('fairway')).toBeInTheDocument();

    view.unmount();
    presetCatalogFailure = true;
    active404Delay = 25;
    render(<TerrainPage />);
    expect(await screen.findByText('Preset catalog unavailable')).toBeInTheDocument();
    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(screen.getByText('Preset catalog unavailable')).toBeInTheDocument();
  });

  it('loads presets with the backend field, normalizes success, and reports failure', async () => {
    terrainLoadResponse = { success: false, error: "Unknown preset 'missing'" };
    render(<TerrainPage />);

    fireEvent.click(await screen.findByRole('button', { name: /putting_green/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Load Terrain' }));

    expect(await screen.findByText(/Unknown preset 'missing'/)).toBeInTheDocument();
    expect(screen.queryByText('loaded_green')).not.toBeInTheDocument();

    terrainLoadResponse = {
      success: true, name: 'loaded_green', width_m: 10, length_m: 15,
      terrain_types: ['green', 'fringe'],
    };
    fireEvent.click(screen.getByRole('button', { name: 'Load Terrain' }));
    expect(await screen.findByText('loaded_green')).toBeInTheDocument();
    const loadCall = mockFetch.mock.calls.find(([input]) => String(input) === '/api/terrain/load');
    expect(JSON.parse(String(loadCall?.[1]?.body))).toEqual({ preset: 'putting_green' });
    expect(within(screen.getByText('Dimensions').parentElement!).getByText('N/A'))
      .toBeInTheDocument();
  });

  it('keeps non-404 active-terrain failures visible', async () => {
    activeTerrainFailure = { status: 500, detail: 'Terrain service failed' };
    render(<TerrainPage />);

    expect(await screen.findByText('Terrain service failed')).toBeInTheDocument();
  });
});

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
    expect(material.friction_coefficient!).toBeGreaterThan(0);
    expect(material.restitution!).toBeLessThanOrEqual(1);
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
    expect(active.properties.dimensions?.[0]).toBe(200);
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

    expect(props.elevation_range![0]).toBeLessThanOrEqual(props.elevation_range![1]);
    expect(props.slope_range![0]).toBeLessThanOrEqual(props.slope_range![1]);
  });

  it('should handle multiple material friction values', () => {
    const materials: TerrainMaterial[] = [
      { id: 'grass', name: 'Grass', friction_coefficient: 0.65, restitution: 0.3, color: '#2d5a27' },
      { id: 'sand', name: 'Sand', friction_coefficient: 0.35, restitution: 0.1, color: '#c2b280' },
      { id: 'water', name: 'Water', friction_coefficient: 0.05, restitution: 0.0, color: '#1a6b8a' },
    ];

    const sorted = [...materials].sort((a, b) => (b.friction_coefficient ?? 0) - (a.friction_coefficient ?? 0));
    expect(sorted[0].id).toBe('grass');
    expect(sorted[2].id).toBe('water');
  });
});
