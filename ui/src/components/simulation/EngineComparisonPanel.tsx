import { GitCompareArrows, Info, AlertTriangle, CircleCheck } from 'lucide-react';
import type { EngineComparisonViewModel } from './engineComparisonViewModel';

interface Props {
  viewModel: EngineComparisonViewModel;
  onToggleEngine: (engineName: string) => void;
  disabled?: boolean;
}

const severityStyles = {
  ok: 'text-emerald-300 border-emerald-600/40 bg-emerald-500/10',
  warning: 'text-amber-300 border-amber-600/40 bg-amber-500/10',
  critical: 'text-red-300 border-red-600/40 bg-red-500/10',
  pending: 'text-gray-300 border-gray-600/40 bg-gray-700/40',
};

function AnnotationIcon({ severity }: { severity: keyof typeof severityStyles }) {
  if (severity === 'ok') {
    return <CircleCheck className="h-3.5 w-3.5" aria-hidden="true" />;
  }
  if (severity === 'warning' || severity === 'critical') {
    return <AlertTriangle className="h-3.5 w-3.5" aria-hidden="true" />;
  }
  return <Info className="h-3.5 w-3.5" aria-hidden="true" />;
}

export function EngineComparisonPanel({ viewModel, onToggleEngine, disabled }: Props) {
  return (
    <section aria-labelledby="engine-comparison-heading">
      <div className="mb-3 flex items-center gap-2">
        <GitCompareArrows className="h-4 w-4 text-blue-300" aria-hidden="true" />
        <h3
          id="engine-comparison-heading"
          className="text-sm font-semibold uppercase tracking-wider text-gray-400"
        >
          Engine Comparison
        </h3>
      </div>

      <div className="mb-3 text-xs text-gray-500">{viewModel.datasetLabel}</div>

      <div className="mb-4 grid grid-cols-1 gap-2" aria-label="Comparison engines">
        {viewModel.options.map((option) => {
          const checked = viewModel.selectedEngineNames.includes(option.name);
          const isDisabled = disabled || option.support !== 'ready';
          return (
            <label
              key={option.name}
              className={`flex min-h-12 items-start gap-2 rounded-md border p-2 text-sm ${
                checked
                  ? 'border-blue-500/60 bg-blue-500/15 text-white'
                  : 'border-gray-700 bg-gray-700/30 text-gray-300'
              } ${isDisabled ? 'opacity-55' : 'cursor-pointer hover:bg-white/5'}`}
            >
              <input
                type="checkbox"
                checked={checked}
                disabled={isDisabled}
                onChange={() => onToggleEngine(option.name)}
                className="mt-1 h-4 w-4 rounded border-gray-500 accent-blue-500"
                aria-label={`Compare ${option.displayName}`}
              />
              <span className="min-w-0">
                <span className="block font-medium">{option.displayName}</span>
                <span className="block truncate text-xs text-gray-500">
                  {option.disabledReason ?? option.capabilities.join(', ')}
                </span>
              </span>
            </label>
          );
        })}
      </div>

      {viewModel.emptyMessage && (
        <div className="mb-3 rounded-md border border-gray-700 bg-gray-900/50 p-3 text-xs text-gray-400">
          {viewModel.emptyMessage}
        </div>
      )}

      {viewModel.columns.length > 0 && (
        <div className="mb-4 grid grid-cols-1 gap-2">
          {viewModel.columns.map((column) => (
            <article
              key={column.name}
              className="rounded-md border border-gray-700 bg-gray-900/50 p-3"
            >
              <div className="flex items-center justify-between gap-2">
                <h4 className="truncate text-sm font-semibold text-white">
                  {column.displayName}
                </h4>
                <span
                  className={`shrink-0 rounded px-2 py-0.5 text-[10px] ${
                    column.hasFrame
                      ? 'bg-emerald-500/15 text-emerald-300'
                      : 'bg-gray-700 text-gray-400'
                  }`}
                >
                  {column.hasFrame ? 'Captured' : 'Pending'}
                </span>
              </div>
              <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                <dt className="text-gray-500">Version</dt>
                <dd className="truncate text-gray-300">{column.provenance.version}</dd>
                <dt className="text-gray-500">Frame</dt>
                <dd className="text-gray-300">
                  {column.provenance.frame ?? 'not run'}
                </dd>
                <dt className="text-gray-500">Time</dt>
                <dd className="text-gray-300">
                  {column.provenance.time === null
                    ? 'not run'
                    : `${column.provenance.time.toFixed(3)}s`}
                </dd>
                <dt className="text-gray-500">Metrics</dt>
                <dd className="text-gray-300">{Object.keys(column.metrics).length}</dd>
              </dl>
            </article>
          ))}
        </div>
      )}

      {viewModel.annotations.length > 0 && (
        <div className="space-y-2" aria-label="Divergence annotations">
          {viewModel.annotations.slice(0, 8).map((annotation) => (
            <div
              key={`${annotation.baseline}-${annotation.compared}-${annotation.metric}`}
              className={`flex items-start gap-2 rounded-md border p-2 text-xs ${
                severityStyles[annotation.severity]
              }`}
            >
              <AnnotationIcon severity={annotation.severity} />
              <span>{annotation.label}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
