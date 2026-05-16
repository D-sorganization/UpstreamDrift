/**
 * Terrain Engine Hook — Fetches terrain configuration, presets, materials,
 * and types from the backend REST API.
 *
 * Provides methods for loading terrain, querying properties, and managing
 * the active terrain session.
 */

import { useState, useCallback, useEffect, useRef } from 'react';

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
  friction_coefficient: number;
  restitution: number;
  color: string;
}

export interface TerrainTypeInfo {
  id: string;
  name: string;
  description: string;
}

export interface TerrainProperties {
  dimensions: [number, number, number];
  resolution: number;
  material: string;
  elevation_range: [number, number];
  slope_range: [number, number];
  features: string[];
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
      const res = await fetch('/api/terrain/presets');
      if (!res.ok) throw new Error(`Failed to fetch presets: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setPresets(data.presets ?? data ?? []);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch presets');
    }
  }, []);

  const fetchMaterials = useCallback(async () => {
    try {
      const res = await fetch('/api/terrain/materials');
      if (!res.ok) throw new Error(`Failed to fetch materials: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setMaterials(data.materials ?? data ?? []);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch materials');
    }
  }, []);

  const fetchTerrainTypes = useCallback(async () => {
    try {
      const res = await fetch('/api/terrain/types');
      if (!res.ok) throw new Error(`Failed to fetch terrain types: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setTerrainTypes(data.types ?? data ?? []);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch terrain types');
    }
  }, []);

  const fetchActiveTerrain = useCallback(async () => {
    try {
      const res = await fetch('/api/terrain/active');
      if (!res.ok && res.status !== 404) throw new Error(`Failed to fetch active terrain: ${res.status}`);
      if (res.status === 404) {
        if (isMountedRef.current) setActiveTerrain(null);
        return;
      }
      const data = await res.json();
      if (isMountedRef.current) setActiveTerrain(data);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Failed to fetch active terrain');
    }
  }, []);

  const loadTerrain = useCallback(async (presetId: string, overrides?: Record<string, unknown>) => {
    setLoadState('loading');
    setError(null);
    try {
      const res = await fetch('/api/terrain/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ preset_id: presetId, ...overrides }),
      });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Failed to load terrain: ${res.status}`);
      }
      const data = await res.json();
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
      const res = await fetch('/api/terrain/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ property }),
      });
      if (!res.ok) throw new Error(`Query failed: ${res.status}`);
      const data = await res.json();
      if (isMountedRef.current) setQueryResult(data);
    } catch (err) {
      if (isMountedRef.current) setError(err instanceof Error ? err.message : 'Terrain query failed');
    }
  }, []);

  // Fetch catalog data on mount
  useEffect(() => {
    isMountedRef.current = true;
    fetchPresets();
    fetchMaterials();
    fetchTerrainTypes();
    fetchActiveTerrain();
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
