/** A labelled range slider with a live numeric readout (CMB-7a, #11658). */
interface SpecSliderProps {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  displayValue: string;
  onChange: (value: number) => void;
}

export function SpecSlider({
  id,
  label,
  min,
  max,
  step,
  value,
  displayValue,
  onChange,
}: SpecSliderProps) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <label htmlFor={id} className="text-xs font-semibold text-gray-300">
          {label}
        </label>
        <span className="text-xs font-mono text-blue-400">{displayValue}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
      />
    </div>
  );
}
