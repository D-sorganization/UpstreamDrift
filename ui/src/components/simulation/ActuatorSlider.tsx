import { useState, useCallback, useEffect, useRef } from 'react';
import { ActuatorInfo } from './ActuatorPanel';

const CONTROL_TYPE_LABELS: Record<string, string> = {
  constant: 'Constant',
  polynomial: 'Polynomial',
  pd_gains: 'PD Gains',
  trajectory: 'Trajectory',
};

export function ActuatorSlider({
  actuator,
  onValueChange,
  onControlTypeChange,
  availableTypes,
}: {
  actuator: ActuatorInfo;
  onValueChange: (index: number, value: number) => void;
  onControlTypeChange: (index: number, type: string) => void;
  availableTypes: string[];
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [dragValue, setDragValue] = useState(actuator.value);
  const localValue = isDragging ? dragValue : actuator.value;

  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  const handleChange = useCallback(
    (newValue: number) => {
      setIsDragging(true);
      setDragValue(newValue);

      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        onValueChange(actuator.index, newValue);
        setIsDragging(false);
      }, 50);
    },
    [actuator.index, onValueChange],
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const range = actuator.max_value - actuator.min_value;
  const percentage =
    range > 0
      ? ((localValue - actuator.min_value) / range) * 100
      : 50;

  return (
    <div className="bg-gray-700/30 p-2 rounded-md">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-mono text-gray-300 truncate max-w-[120px]">
          {actuator.name}
        </span>
        <select
          value={actuator.control_type}
          onChange={(e) =>
            onControlTypeChange(actuator.index, e.target.value)
          }
          className="text-xs bg-gray-600 text-gray-300 rounded px-1 py-0.5 border-none focus:ring-1 focus:ring-blue-400"
        >
          {availableTypes.map((type) => (
            <option key={type} value={type}>
              {CONTROL_TYPE_LABELS[type] || type}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 w-12 text-right font-mono">
          {actuator.min_value.toFixed(1)}
        </span>
        <div className="flex-1 relative">
          <input
            type="range"
            min={actuator.min_value}
            max={actuator.max_value}
            step={(range / 200) || 0.1}
            value={localValue}
            onChange={(e) => handleChange(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-gray-600 rounded-lg appearance-none cursor-pointer"
            aria-label={`${actuator.name} control value`}
          />
          <div
            className="absolute top-0 left-0 h-1.5 bg-blue-500 rounded-l-lg pointer-events-none"
            style={{ width: `${Math.max(0, Math.min(100, percentage))}%` }}
          />
        </div>
        <span className="text-xs text-gray-500 w-12 font-mono">
          {actuator.max_value.toFixed(1)}
        </span>
      </div>

      <div className="flex items-center justify-between mt-1">
        <span className="text-xs text-gray-400">
          {actuator.joint_type}
        </span>
        <span className="text-xs font-mono text-blue-400">
          {localValue.toFixed(2)} {actuator.units}
        </span>
      </div>
    </div>
  );
}
