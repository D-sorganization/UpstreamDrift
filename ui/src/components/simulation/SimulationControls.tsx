import { Play, Pause, Square } from 'lucide-react';

interface Props {
  isRunning: boolean;
  isPaused?: boolean;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
}

export function SimulationControls({ isRunning, isPaused, onStart, onStop, onPause, onResume }: Props) {
  return (
    <div
      className="flex gap-2 mt-4"
      role="toolbar"
      aria-label="Simulation controls"
    >
      {!isRunning ? (
        <button
          onClick={onStart}
          aria-label="Start simulation"
          className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-offset-2 focus:ring-offset-gray-900"
        >
          <Play size={20} aria-hidden="true" />
          Start
        </button>
      ) : (
        <>
            {isPaused ? (
                 <button
                    onClick={onResume}
                    aria-label="Resume simulation"
                    className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-offset-2 focus:ring-offset-gray-900"
                >
                    <Play size={20} aria-hidden="true" />
                    Resume
                </button>
            ) : (
                 <button
                    onClick={onPause}
                    aria-label="Pause simulation"
                    className="flex-1 flex items-center justify-center gap-2 bg-yellow-600 hover:bg-yellow-700 text-white font-semibold py-2 px-4 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-yellow-400 focus:ring-offset-2 focus:ring-offset-gray-900"
                >
                    <Pause size={20} aria-hidden="true" />
                    Pause
                </button>
            )}

          <button
            onClick={onStop}
            aria-label="Stop simulation"
            className="flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-offset-2 focus:ring-offset-gray-900"
          >
            <Square size={20} aria-hidden="true" />
            Stop
          </button>
        </>
      )}
    </div>
  );
}
