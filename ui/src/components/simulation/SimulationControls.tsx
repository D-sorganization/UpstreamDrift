import { Play, Pause, Square } from 'lucide-react';

interface Props {
  isRunning: boolean;
  isPaused?: boolean;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  disabled?: boolean;
}

// Extracted button styles to reduce duplication and ensure consistency
const buttonStyles = {
  base: 'flex items-center justify-center gap-2 font-semibold py-2 px-4 rounded transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-900',
  primary: 'flex-1 bg-green-600 hover:bg-green-700 text-white focus:ring-green-400',
  warning: 'flex-1 bg-yellow-600 hover:bg-yellow-700 text-white focus:ring-yellow-400',
  danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-400',
  disabled: 'flex-1 bg-gray-600 text-gray-400 cursor-not-allowed',
};

export function SimulationControls({ isRunning, isPaused, onStart, onStop, onPause, onResume, disabled }: Props) {
  return (
    <div
      className="flex gap-2 mt-4"
      role="toolbar"
      aria-label="Simulation controls"
    >
      {!isRunning ? (
        <button
          onClick={onStart}
          disabled={disabled}
          aria-label="Start simulation"
          className={`${buttonStyles.base} ${disabled ? buttonStyles.disabled : buttonStyles.primary}`}
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
              className={`${buttonStyles.base} ${buttonStyles.primary}`}
            >
              <Play size={20} aria-hidden="true" />
              Resume
            </button>
          ) : (
            <button
              onClick={onPause}
              aria-label="Pause simulation"
              className={`${buttonStyles.base} ${buttonStyles.warning}`}
            >
              <Pause size={20} aria-hidden="true" />
              Pause
            </button>
          )}

          <button
            onClick={onStop}
            aria-label="Stop simulation"
            className={`${buttonStyles.base} ${buttonStyles.danger}`}
          >
            <Square size={20} aria-hidden="true" />
            Stop
          </button>
        </>
      )}
    </div>
  );
}
