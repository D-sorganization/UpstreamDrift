import { useQuery } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { fetchEngines } from '@/api/client';
import { isTauri, startBackend } from '@/api/backend';

interface Props {
  value: string;
  onChange: (engine: string) => void;
  disabled?: boolean;
}

export function EngineSelector({ value, onChange, disabled }: Props) {
  const autoStartAttempted = useRef(false);

  const { data: engines, isLoading, error, refetch } = useQuery({
    queryKey: ['engines'],
    queryFn: fetchEngines,
    retry: 3,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10000),
  });

  // Auto-start backend in Tauri if engine fetch fails
  useEffect(() => {
    if (error && isTauri() && !autoStartAttempted.current) {
      autoStartAttempted.current = true;
      startBackend()
        .then(() => {
          // Wait for server to initialize, then retry
          setTimeout(() => refetch(), 3000);
        })
        .catch((err) => {
          console.error('Failed to auto-start backend:', err);
        });
    }
  }, [error, refetch]);

  if (isLoading) {
    return (
      <div
        className="animate-pulse h-10 bg-gray-700 rounded"
        role="status"
        aria-label="Loading physics engines"
      >
        <span className="sr-only">Loading engines...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Physics Engine
        </label>
        <div
          className="p-4 rounded-lg border border-amber-500 bg-amber-500/10 text-amber-300"
          role="alert"
          aria-live="polite"
        >
          <div className="font-medium mb-2">Backend server not connected</div>
          <div className="text-xs mb-3 text-amber-400/80">
            {isTauri()
              ? 'The Python backend is starting up. If this persists, check the diagnostics panel (bottom-right).'
              : 'Start the backend server: python launch_golf_suite.py --api-only'}
          </div>
          <button
            onClick={() => {
              if (isTauri()) {
                autoStartAttempted.current = false;
                startBackend().then(() => setTimeout(() => refetch(), 3000));
              } else {
                refetch();
              }
            }}
            aria-label="Retry loading physics engines"
            className="px-3 py-1 text-xs bg-amber-500/20 border border-amber-500 rounded hover:bg-amber-500/30 transition-colors focus:outline-none focus:ring-2 focus:ring-amber-400 focus:ring-offset-2 focus:ring-offset-gray-900"
          >
            {isTauri() ? 'Start Backend & Retry' : 'Retry'}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-6">
      <label
        id="engine-selector-label"
        className="block text-sm font-medium text-gray-300 mb-2"
      >
        Physics Engine
      </label>
      <div
        className="grid grid-cols-1 gap-2"
        role="radiogroup"
        aria-labelledby="engine-selector-label"
      >
        {engines?.map((engine, index) => (
          <button
            key={`${engine.name}-${index}`}
            onClick={() => onChange(engine.name)}
            disabled={disabled || !engine.available}
            role="radio"
            aria-checked={value === engine.name}
            aria-label={`${engine.name} physics engine${!engine.available ? ' (not installed)' : ''}`}
            className={`
              p-3 rounded-lg border text-left transition-all
              ${value === engine.name
                ? 'border-blue-500 bg-blue-500/20 text-white'
                : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
              }
              ${!engine.available && 'opacity-50 cursor-not-allowed'}
              ${disabled && 'cursor-not-allowed'}
              focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-900
            `}
          >
            <div className="font-medium">{engine.name}</div>
            <div className="text-xs text-gray-400">
              {engine.available ? (
                engine.loaded ? '● Loaded' : '○ Available'
              ) : (
                '✗ Not installed'
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
