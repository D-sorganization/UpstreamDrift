import { useState } from 'react';
import type { ActuatorInfo } from './ActuatorPanel';

interface PolynomialGeneratorPanelProps {
  actuators: ActuatorInfo[];
  onApplyPolynomial: (actuatorIndex: number, coefficients: number[]) => Promise<void>;
}

export function PolynomialGeneratorPanel({
  actuators,
  onApplyPolynomial,
}: PolynomialGeneratorPanelProps) {
  const [collapsed, setCollapsed] = useState(true);
  const [selectedActuatorIndex, setSelectedActuatorIndex] = useState(
    actuators.length > 0 ? actuators[0].index : 0
  );
  // Stored as raw strings so empty/NaN input is detectable (not silently
  // coerced to 0) — see issue #7429.
  const [coefficients, setCoefficients] = useState<string[]>(['1', '0']);

  if (actuators.length === 0) {
    return null;
  }

  const parsed = coefficients.map((c) => Number(c));
  const hasInvalid = coefficients.some(
    (c) => c.trim() === '' || Number.isNaN(Number(c))
  );
  // A polynomial needs at least one term and every term must be valid.
  const canApply = coefficients.length > 0 && !hasInvalid;
  const canRemove = coefficients.length > 1;

  const handleApply = () => {
    if (!canApply) {
      return;
    }
    void onApplyPolynomial(selectedActuatorIndex, parsed);
  };

  return (
    <div className="bg-gray-800/40 border border-white/5 rounded-md p-3">
      {/* Header Button */}
      <button
        type="button"
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-between w-full text-left font-semibold text-xs text-gray-300 uppercase tracking-wider"
      >
        <span>Polynomial Generator</span>
        <span className="text-gray-500">{collapsed ? '+' : '-'}</span>
      </button>

      {!collapsed && (
        <div className="mt-3 space-y-3">
          {/* Actuator Selector */}
          <div className="flex flex-col gap-1">
            <label htmlFor="poly-actuator-select" className="text-xs text-gray-400">
              Actuator
            </label>
            <select
              id="poly-actuator-select"
              value={selectedActuatorIndex}
              onChange={(e) => setSelectedActuatorIndex(parseInt(e.target.value, 10))}
              className="text-xs bg-gray-700 text-gray-200 rounded px-2 py-1.5 border-none focus:ring-1 focus:ring-blue-400"
            >
              {actuators.map((act) => (
                <option key={act.index} value={act.index}>
                  {act.name}
                </option>
              ))}
            </select>
          </div>

          {/* Coefficients List */}
          <div className="space-y-2">
            <span className="text-xs text-gray-400 block font-medium">Coefficients</span>
            {coefficients.map((coeff, index) => {
              const invalid = coeff.trim() === '' || Number.isNaN(Number(coeff));
              return (
                <div key={index} className="flex items-center gap-2">
                  <label
                    htmlFor={`coeff-${index}`}
                    className="text-xs text-gray-500 w-24"
                  >
                    Coefficient {index}
                  </label>
                  <input
                    id={`coeff-${index}`}
                    type="number"
                    step="0.1"
                    value={coeff}
                    onChange={(e) => {
                      const next = [...coefficients];
                      next[index] = e.target.value;
                      setCoefficients(next);
                    }}
                    aria-invalid={invalid}
                    className={`w-20 text-xs bg-gray-700 text-gray-200 rounded px-1.5 py-1 focus:ring-1 focus:ring-blue-400 font-mono ${
                      invalid ? 'border border-red-500' : 'border-none'
                    }`}
                    aria-label={`Coefficient ${index}`}
                  />
                  <button
                    type="button"
                    disabled={!canRemove}
                    title={
                      canRemove
                        ? undefined
                        : 'A polynomial needs at least one coefficient'
                    }
                    onClick={() => {
                      if (!canRemove) {
                        return;
                      }
                      const next = coefficients.filter((_, i) => i !== index);
                      setCoefficients(next);
                    }}
                    className="text-xs text-red-400 hover:text-red-300 disabled:text-gray-600 disabled:hover:text-gray-600 disabled:cursor-not-allowed font-medium px-2 py-1 rounded hover:bg-red-500/10 transition-colors"
                    aria-label="Remove"
                  >
                    Remove
                  </button>
                </div>
              );
            })}
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-2 border-t border-white/5">
            <button
              type="button"
              onClick={() => setCoefficients([...coefficients, '0'])}
              className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 px-2.5 py-1 rounded transition-colors"
            >
              Add Coefficient
            </button>
            <button
              type="button"
              disabled={!canApply}
              title={canApply ? undefined : 'Fix invalid coefficient'}
              onClick={handleApply}
              className="text-xs bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed text-white font-medium px-3 py-1 rounded transition-colors ml-auto"
            >
              Apply Polynomial
            </button>
          </div>
          {hasInvalid && (
            <p role="alert" className="text-xs text-red-400">
              Fix invalid coefficient before applying.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
