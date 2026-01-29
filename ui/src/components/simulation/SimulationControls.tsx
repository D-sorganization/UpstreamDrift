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
    <div className="flex gap-2 mt-4">
      {!isRunning ? (
        <button
          onClick={onStart}
          className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded transition-colors"
        >
          <Play size={20} />
          Start
        </button>
      ) : (
        <>
            {isPaused ? (
                 <button
                    onClick={onResume}
                    className="flex-1 flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded transition-colors"
                >
                    <Play size={20} />
                    Resume
                </button>
            ) : (
                 <button
                    onClick={onPause}
                    className="flex-1 flex items-center justify-center gap-2 bg-yellow-600 hover:bg-yellow-700 text-white font-semibold py-2 px-4 rounded transition-colors"
                >
                    <Pause size={20} />
                    Pause
                </button>
            )}
            
          <button
            onClick={onStop}
            className="flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded transition-colors"
          >
            <Square size={20} />
            Stop
          </button>
        </>
      )}
    </div>
  );
}
