/**
 * SummaryPanel - Displays a RunSummary after simulation completes.
 *
 * Renders engine name, duration, step count, peak torques, energy, and
 * next-step suggestions returned by the backend simulation_ws route.
 *
 * See issue #3174
 */

/** Shape of the RunSummary returned in the WebSocket "complete" message. */
export interface RunSummary {
  engine: string;
  duration_s: number;
  steps: number;
  max_torques: number[];
  energy: number;
  trajectory: Record<string, unknown>[];
  next_steps: string[];
}

interface Props {
  summary: RunSummary;
  onDismiss?: () => void;
}

/** Format a number to at most 3 significant decimal places. */
function fmt(n: number): string {
  return n.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

export function SummaryPanel({ summary, onDismiss }: Props) {
  const peakTorque =
    summary.max_torques.length > 0 ? Math.max(...summary.max_torques) : null;

  return (
    <div
      className="bg-gray-800 border border-gray-600 rounded-lg p-4 space-y-4"
      role="region"
      aria-label="Post-simulation summary"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-green-400 uppercase tracking-wider">
          Run Complete
        </h3>
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="text-gray-500 hover:text-gray-300 text-xs focus:outline-none
                       focus:ring-1 focus:ring-gray-400 rounded"
            aria-label="Dismiss summary"
          >
            ✕
          </button>
        )}
      </div>

      {/* Key metrics grid */}
      <dl className="grid grid-cols-2 gap-3">
        <div className="bg-gray-700/50 p-2 rounded">
          <dt className="text-xs text-gray-400">Engine</dt>
          <dd className="text-sm font-mono text-white capitalize">{summary.engine}</dd>
        </div>
        <div className="bg-gray-700/50 p-2 rounded">
          <dt className="text-xs text-gray-400">Duration</dt>
          <dd className="text-sm font-mono text-white">{fmt(summary.duration_s)} s</dd>
        </div>
        <div className="bg-gray-700/50 p-2 rounded">
          <dt className="text-xs text-gray-400">Steps</dt>
          <dd className="text-sm font-mono text-white">{summary.steps.toLocaleString()}</dd>
        </div>
        <div className="bg-gray-700/50 p-2 rounded">
          <dt className="text-xs text-gray-400">Energy</dt>
          <dd className="text-sm font-mono text-white">{fmt(summary.energy)} J</dd>
        </div>
        {peakTorque !== null && (
          <div className="bg-gray-700/50 p-2 rounded col-span-2">
            <dt className="text-xs text-gray-400">Peak Torque</dt>
            <dd className="text-sm font-mono text-white">{fmt(peakTorque)} N·m</dd>
          </div>
        )}
      </dl>

      {/* Trajectory sample count */}
      {summary.trajectory.length > 0 && (
        <p className="text-xs text-gray-500">
          {summary.trajectory.length} trajectory sample
          {summary.trajectory.length !== 1 ? 's' : ''} recorded
        </p>
      )}

      {/* Next steps */}
      {summary.next_steps.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Suggestions
          </h4>
          <ul className="space-y-1">
            {summary.next_steps.map((step, i) => (
              <li key={i} className="text-xs text-gray-300 flex gap-2">
                <span className="text-blue-400 flex-shrink-0">›</span>
                {step}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
