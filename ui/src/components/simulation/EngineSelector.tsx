import { useQuery } from '@tanstack/react-query';
import { fetchEngines } from '@/api/client';

interface Props {
  value: string;
  onChange: (engine: string) => void;
  disabled?: boolean;
}

export function EngineSelector({ value, onChange, disabled }: Props) {
  const { data: engines, isLoading, error, refetch } = useQuery({
    queryKey: ['engines'],
    queryFn: fetchEngines,
  });

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
          className="p-4 rounded-lg border border-red-500 bg-red-500/10 text-red-400"
          role="alert"
          aria-live="polite"
        >
          <div className="font-medium mb-2">Failed to load engines</div>
          <div className="text-xs mb-3">
            {error instanceof Error ? error.message : 'Unknown error occurred'}
          </div>
          <button
            onClick={() => refetch()}
            aria-label="Retry loading physics engines"
            className="px-3 py-1 text-xs bg-red-500/20 border border-red-500 rounded hover:bg-red-500/30 transition-colors focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-offset-2 focus:ring-offset-gray-900"
          >
            Retry
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
