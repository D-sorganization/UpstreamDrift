import { Loader2, Check, AlertCircle, Power, PowerOff } from 'lucide-react';
import type { ManagedEngine, EngineLoadState } from '@/api/useEngineManager';

interface Props {
  engines: ManagedEngine[];
  selectedEngine: string | null;
  onSelect: (engine: string) => void;
  onLoad: (engine: string) => void;
  onUnload: (engine: string) => void;
  disabled?: boolean;
}

function LoadStateIcon({ state }: { state: EngineLoadState }) {
  switch (state) {
    case 'loading':
      return <Loader2 className="w-4 h-4 animate-spin text-blue-400" aria-hidden="true" />;
    case 'loaded':
      return <Check className="w-4 h-4 text-emerald-400" aria-hidden="true" />;
    case 'error':
      return <AlertCircle className="w-4 h-4 text-red-400" aria-hidden="true" />;
    default:
      return <Power className="w-4 h-4 text-gray-500" aria-hidden="true" />;
  }
}

function LoadStateLabel({ engine }: { engine: ManagedEngine }) {
  switch (engine.loadState) {
    case 'loading':
      return <span className="text-blue-400">Loading...</span>;
    case 'loaded':
      return (
        <span className="text-emerald-400">
          Loaded{engine.version ? ` (v${engine.version})` : ''}
        </span>
      );
    case 'error':
      return <span className="text-red-400">{engine.error || 'Failed to load'}</span>;
    default:
      return <span className="text-gray-500">Not loaded</span>;
  }
}

export function EngineSelector({
  engines,
  selectedEngine,
  onSelect,
  onLoad,
  onUnload,
  disabled,
}: Props) {
  return (
    <div className="mb-6">
      <label
        id="engine-selector-label"
        className="block text-sm font-medium text-gray-300 mb-2"
      >
        Physics Engines
      </label>
      <p className="text-xs text-gray-500 mb-3">
        Click <strong>Load</strong> to activate an engine. Select a loaded engine to simulate.
      </p>
      <div
        className="grid grid-cols-1 gap-2"
        role="radiogroup"
        aria-labelledby="engine-selector-label"
      >
        {engines.map((engine) => {
          const isSelected = selectedEngine === engine.name;
          const isLoaded = engine.loadState === 'loaded';
          const isLoading = engine.loadState === 'loading';
          const isError = engine.loadState === 'error';
          const isIdle = engine.loadState === 'idle';

          return (
            <div
              key={engine.name}
              className={`
                relative rounded-lg border transition-all
                ${isSelected && isLoaded
                  ? 'border-blue-500 bg-blue-500/20 ring-1 ring-blue-500/50'
                  : isLoaded
                    ? 'border-emerald-600/50 bg-emerald-500/10'
                    : isError
                      ? 'border-red-600/50 bg-red-500/5'
                      : 'border-gray-600 bg-gray-700/50'
                }
              `}
            >
              {/* Engine info — clickable to select (only if loaded) */}
              <button
                onClick={() => isLoaded && onSelect(engine.name)}
                disabled={disabled || !isLoaded}
                role="radio"
                aria-checked={isSelected}
                aria-label={`${engine.displayName} physics engine${!isLoaded ? ' (not loaded)' : ''}`}
                className={`
                  w-full text-left p-3 pr-20 rounded-lg transition-colors
                  ${isLoaded ? 'cursor-pointer hover:bg-white/5' : 'cursor-default'}
                  ${disabled ? 'cursor-not-allowed opacity-60' : ''}
                  focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-offset-2 focus:ring-offset-gray-900
                `}
              >
                <div className="flex items-center gap-2">
                  <LoadStateIcon state={engine.loadState} />
                  <span className={`font-medium ${isLoaded ? 'text-white' : 'text-gray-300'}`}>
                    {engine.displayName}
                  </span>
                </div>
                <div className="text-xs text-gray-400 mt-1 ml-6">
                  {engine.description}
                </div>
                <div className="text-xs mt-1 ml-6">
                  <LoadStateLabel engine={engine} />
                </div>
              </button>

              {/* Load/Unload button — absolute positioned on right */}
              <div className="absolute right-2 top-1/2 -translate-y-1/2">
                {isLoaded ? (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onUnload(engine.name);
                    }}
                    disabled={disabled || isSelected}
                    aria-label={`Unload ${engine.displayName}`}
                    title={isSelected ? 'Cannot unload active engine' : `Unload ${engine.displayName}`}
                    className={`
                      p-2 rounded-md text-xs transition-colors
                      ${isSelected
                        ? 'text-gray-600 cursor-not-allowed'
                        : 'text-gray-400 hover:text-red-400 hover:bg-red-500/10'
                      }
                      focus:outline-none focus:ring-2 focus:ring-red-400
                    `}
                  >
                    <PowerOff className="w-4 h-4" />
                  </button>
                ) : (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onLoad(engine.name);
                    }}
                    disabled={disabled || isLoading}
                    aria-label={`Load ${engine.displayName} engine`}
                    className={`
                      px-3 py-1.5 rounded-md text-xs font-medium transition-all
                      ${isLoading
                        ? 'bg-blue-500/20 text-blue-400 cursor-wait'
                        : isError
                          ? 'bg-red-500/20 text-red-300 hover:bg-red-500/30 border border-red-600/50'
                          : 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30 border border-emerald-600/50'
                      }
                      ${disabled ? 'cursor-not-allowed opacity-60' : ''}
                      focus:outline-none focus:ring-2 focus:ring-emerald-400
                    `}
                  >
                    {isLoading ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : isError ? (
                      'Retry'
                    ) : (
                      'Load'
                    )}
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
