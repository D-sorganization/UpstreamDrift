/**
 * Hook to fetch a URDF model from the backend API.
 *
 * Returns the model data and loading/error states.
 *
 * See issue #1201
 */

import { useState, useCallback, useEffect } from 'react';
import type { URDFModel } from '@/components/visualization/URDFViewer';

export function useURDFModel(modelName: string | null) {
  const [model, setModel] = useState<URDFModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchModel = useCallback(async (name: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(name)}/urdf`);
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Failed to load model: ${response.status}`);
      }
      const data = await response.json();
      setModel(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setModel(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (modelName) {
      fetchModel(modelName);
    } else {
      setModel(null);
    }
  }, [modelName, fetchModel]);

  return { model, loading, error, refetch: fetchModel };
}
