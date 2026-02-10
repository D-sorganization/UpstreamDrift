/**
 * Engine Capabilities Hook - Dynamic UI Adaptation
 *
 * Fetches and caches engine capability data from the backend,
 * enabling the UI to dynamically show/hide features based on
 * what the active physics engine supports.
 *
 * See issue #1204
 */

import { useState, useCallback, useEffect, useRef } from 'react';

/** Support level for a single engine capability. */
export type CapabilityLevel = 'full' | 'partial' | 'none';

/** A single capability entry from the backend. */
export interface CapabilityEntry {
  name: string;
  level: CapabilityLevel;
  supported: boolean;
}

/** Full capabilities response from the backend. */
export interface EngineCapabilitiesData {
  engine_name: string;
  engine_type: string;
  capabilities: CapabilityEntry[];
  summary: {
    full: number;
    partial: number;
    none: number;
  };
}

export type CapabilitiesLoadState = 'idle' | 'loading' | 'loaded' | 'error';

export interface UseEngineCapabilitiesResult {
  /** Current capabilities data, null if not loaded. */
  capabilities: EngineCapabilitiesData | null;
  /** Loading state. */
  loadState: CapabilitiesLoadState;
  /** Error message if loading failed. */
  error: string | null;
  /** Fetch capabilities for a specific engine. */
  fetchCapabilities: (engineType: string) => Promise<void>;
  /** Check if a specific capability is supported (full or partial). */
  isSupported: (capabilityName: string) => boolean;
  /** Check if a capability has full support. */
  isFullySupported: (capabilityName: string) => boolean;
  /** Get the level of a specific capability. */
  getLevel: (capabilityName: string) => CapabilityLevel;
}

/**
 * Hook to fetch and query engine capabilities.
 *
 * Automatically fetches capabilities when engineType changes.
 * Provides helper methods for checking feature support levels.
 *
 * @param engineType - Engine type to fetch capabilities for.
 *                     If undefined, capabilities are not fetched.
 */
export function useEngineCapabilities(
  engineType?: string,
): UseEngineCapabilitiesResult {
  const [capabilities, setCapabilities] = useState<EngineCapabilitiesData | null>(null);
  const [loadState, setLoadState] = useState<CapabilitiesLoadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef(true);

  const fetchCapabilities = useCallback(async (engine: string) => {
    setLoadState('loading');
    setError(null);

    try {
      const response = await fetch(`/engines/${engine}/capabilities`);
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Failed to fetch capabilities for ${engine}`);
      }

      const data: EngineCapabilitiesData = await response.json();

      if (isMountedRef.current) {
        setCapabilities(data);
        setLoadState('loaded');
      }
    } catch (err) {
      if (isMountedRef.current) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        setError(message);
        setLoadState('error');
      }
    }
  }, []);

  // Auto-fetch when engineType changes
  useEffect(() => {
    if (engineType) {
      fetchCapabilities(engineType);
    } else {
      setCapabilities(null);
      setLoadState('idle');
      setError(null);
    }
  }, [engineType, fetchCapabilities]);

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const isSupported = useCallback(
    (capabilityName: string): boolean => {
      if (!capabilities) return false;
      const cap = capabilities.capabilities.find((c) => c.name === capabilityName);
      return cap?.supported ?? false;
    },
    [capabilities],
  );

  const isFullySupported = useCallback(
    (capabilityName: string): boolean => {
      if (!capabilities) return false;
      const cap = capabilities.capabilities.find((c) => c.name === capabilityName);
      return cap?.level === 'full';
    },
    [capabilities],
  );

  const getLevel = useCallback(
    (capabilityName: string): CapabilityLevel => {
      if (!capabilities) return 'none';
      const cap = capabilities.capabilities.find((c) => c.name === capabilityName);
      return cap?.level ?? 'none';
    },
    [capabilities],
  );

  return {
    capabilities,
    loadState,
    error,
    fetchCapabilities,
    isSupported,
    isFullySupported,
    getLevel,
  };
}
