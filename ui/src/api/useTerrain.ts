/**
 * Terrain Engine Hook — Fetches terrain configuration, presets, materials,
 * and types from the backend REST API.
 *
 * Provides methods for loading terrain, querying properties, and managing
 * the active terrain session.
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { ApiHttpError, apiFetch, apiFetchParsed } from './fetch';
import {
  parseActiveTerrain,
  parseTerrainMaterials,
  parseTerrainLoadResponse,
  parseTerrainPresets,
  parseTerrainTypes,
} from './schemas';

// ── Types ──────────────────────────────────────────────────────────────

export interface TerrainPreset {
  id: string;
  name: string;
  description: string;
  terrain_type: string;
  defaults: Record<string, unknown>;
}

export interface TerrainMaterial {
  id: string;
  name: string;
  friction_coefficient?: number;
  restitution?: number;
  color?: string;
}

export interface TerrainTypeInfo {
  id: string;
  name: string;
  description: string;
}

export interface TerrainProperties {
  dimensions?: number[];
  resolution?: number;
  material?: string;
  elevation_range?: number[];
  slope_range?: number[];
  features?: string[];
}

export interface ActiveTerrain {
  id: string;
  name: string;
  terrain_type: string;
  material: string;
  properties: TerrainProperties;
  loaded_at: string;
}

export type TerrainLoadState = 'idle' | 'loading' | 'loaded' | 'error';

// ── Hook ───────────────────────────────────────────────────────────────

export function useTerrain() {
  const [presets, setPresets] = useState<TerrainPreset[]>([]);
  const [materials, setMaterials] = useState<TerrainMaterial[]>([]);
  const [terrainTypes, setTerrainTypes] = useState<TerrainTypeInfo[]>([]);
  const [activeTerrain, setActiveTerrain] = useState<ActiveTerrain | null>(null);
  const [queryResult, setQueryResult] = useState<TerrainProperties | null>(null);
  const [loadState, setLoadState] = useState<TerrainLoadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  const fetchPresets = useCallback(async () => {
    try {
      const data = await apiFetchParsed('/api/terrain/presets', parseTerrainPresets);
      if (isMountedRef.current) setPresets(data);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch presets');
    }
  }, []);

  const fetchMaterials = useCallback(async () => {
    try {
      const data = await apiFetchParsed('/api/terrain/materials', parseTerrainMaterials);
      if (isMountedRef.current) setMaterials(data);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch materials');
    }
  }, []);

  const fetchTerrainTypes = useCallback(async () => {
    try {
      const data = await apiFetchParsed('/api/terrain/types', parseTerrainTypes);
      if (isMountedRef.current) setTerrainTypes(data);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch terrain types');
    }
  }, []);

  const fetchActiveTerrain = useCallback(async () => {
    try {
      const data = await apiFetchParsed('/api/terrain/active', parseActiveTerrain);
      if (isMountedRef.current) setActiveTerrain(data);
    } catch (err) {
      // A 404 means no terrain is loaded yet — not an error to surface.
      const message = err instanceof Error ? err.message : 'Failed to fetch active terrain';
      if (err instanceof ApiHttpError && err.status === 404) {
        if (isMountedRef.current) {
          setActiveTerrain(null);
        }
        return;
      }
      if (isMountedRef.current) setError(message);
    }
  }, []);

  const loadTerrain = useCallback(async (presetId: string, overrides?: Record<string, unknown>) => {
    setLoadState('loading');
    setError(null);
    try {
      const data = await apiFetchParsed('/api/terrain/load', parseTerrainLoadResponse, {
        method: 'POST',
        body: JSON.stringify({ ...overrides, preset: presetId }),
      });
      if (isMountedRef.current) {
        setActiveTerrain(data);
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to load terrain');
        setLoadState('error');
      }
    }
  }, []);

  const queryTerrain = useCallback(async (property: string) => {
    try {
      const data = await apiFetch<TerrainProperties>('/api/terrain/query', {
        method: 'POST',
        body: JSON.stringify({ property }),
      });
      if (isMountedRef.current) setQueryResult(data);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Terrain query failed');
    }
  }, []);

  // Fetch catalog data on mount. These fetchers are async (they await apiFetch
  // before any setState); scheduling via a microtask makes the deferral
  // explicit for react-hooks/set-state-in-effect.
  useEffect(() => {
    isMountedRef.current = true;
    void Promise.resolve().then(() => {
      fetchPresets();
      fetchMaterials();
      fetchTerrainTypes();
      fetchActiveTerrain();
    });
    return () => { isMountedRef.current = false; };
  }, [fetchPresets, fetchMaterials, fetchTerrainTypes, fetchActiveTerrain]);

  return {
    presets,
    materials,
    terrainTypes,
    activeTerrain,
    queryResult,
    loadState,
    error,
    loadTerrain,
    queryTerrain,
    refetch: () => { fetchPresets(); fetchMaterials(); fetchTerrainTypes(); fetchActiveTerrain(); },
  };
}
