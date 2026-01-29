import { useQuery } from '@tanstack/react-query';
import { fetchEngines } from '../../api/client';

interface Props {
  value: string;
  onChange: (engine: string) => void;
  disabled?: boolean;
}

export function EngineSelector({ value, onChange, disabled }: Props) {
  const { data: engines, isLoading } = useQuery({
    queryKey: ['engines'],
    queryFn: fetchEngines,
  });

  if (isLoading) {
    return <div className="animate-pulse h-10 bg-gray-700 rounded" />;
  }

  return (
    <div className="mb-6">
      <label className="block text-sm font-medium text-gray-300 mb-2">
        Physics Engine
      </label>
      <div className="grid grid-cols-1 gap-2">
        {engines?.map((engine) => (
          <button
            key={engine.name}
            onClick={() => onChange(engine.name)}
            disabled={disabled || !engine.available}
            className={`
              p-3 rounded-lg border text-left transition-all
              ${value === engine.name
                ? 'border-blue-500 bg-blue-500/20 text-white'
                : 'border-gray-600 bg-gray-700 text-gray-300 hover:border-gray-500'
              }
              ${!engine.available && 'opacity-50 cursor-not-allowed'}
              ${disabled && 'cursor-not-allowed'}
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
