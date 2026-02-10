/**
 * ForceOverlayPanel - UI controls for force/torque overlay configuration.
 *
 * Provides toggles for force types, body filtering, magnitude color-coding,
 * and label display. Fetches overlay data from the backend.
 *
 * See issue #1199
 */

import { useState, useCallback, useEffect } from 'react';
import type { ForceVector3D, ForceOverlayConfig } from './ForceOverlay';

interface ForceOverlayPanelProps {
  /** Callback when vectors update */
  onVectorsChange: (vectors: ForceVector3D[]) => void;
  /** Whether simulation is running */
  isRunning: boolean;
  /** Polling interval in ms (0 to disable) */
  pollInterval?: number;
}

const DEFAULT_CONFIG: ForceOverlayConfig = {
  enabled: false,
  forceTypes: ['applied'],
  scaleFactor: 0.01,
  colorByMagnitude: true,
  showLabels: false,
  bodyFilter: null,
};

const FORCE_TYPE_OPTIONS = [
  { value: 'applied', label: 'Applied Torques', color: 'text-red-400' },
  { value: 'gravity', label: 'Gravity', color: 'text-blue-400' },
  { value: 'contact', label: 'Contact Forces', color: 'text-green-400' },
  { value: 'bias', label: 'Bias Forces', color: 'text-yellow-400' },
];

/**
 * ForceOverlayPanel provides UI controls for configuring force overlays
 * and polls the backend for updated vector data.
 *
 * See issue #1199
 */
export function ForceOverlayPanel({
  onVectorsChange,
  isRunning,
  pollInterval = 200,
}: ForceOverlayPanelProps) {
  const [config, setConfig] = useState<ForceOverlayConfig>(DEFAULT_CONFIG);
  const [totalForce, setTotalForce] = useState(0);
  const [totalTorque, setTotalTorque] = useState(0);

  const fetchVectors = useCallback(async () => {
    if (!config.enabled) {
      onVectorsChange([]);
      return;
    }

    try {
      const params = new URLSearchParams({
        force_types: config.forceTypes.join(','),
        color_by_magnitude: String(config.colorByMagnitude),
        show_labels: String(config.showLabels),
        scale_factor: String(config.scaleFactor),
      });
      if (config.bodyFilter) {
        params.set('body_filter', config.bodyFilter.join(','));
      }

      const response = await fetch(`/api/simulation/forces?${params}`);
      if (!response.ok) return;

      const data = await response.json();
      onVectorsChange(data.vectors || []);
      setTotalForce(data.total_force_magnitude || 0);
      setTotalTorque(data.total_torque_magnitude || 0);
    } catch {
      // Silently fail on network errors during polling
    }
  }, [config, onVectorsChange]);

  // Poll for vectors when enabled and simulation is running.
  // Use setInterval for polling; the callback (fetchVectors) calls
  // setState asynchronously via the fetch callback, not synchronously
  // in the effect body.
  useEffect(() => {
    if (!config.enabled || !isRunning || pollInterval <= 0) {
      return;
    }

    const interval = setInterval(fetchVectors, pollInterval);
    return () => clearInterval(interval);
  }, [config.enabled, isRunning, pollInterval, fetchVectors]);

  const toggleForceType = useCallback((forceType: string) => {
    setConfig((prev) => {
      const types = prev.forceTypes.includes(forceType)
        ? prev.forceTypes.filter((t) => t !== forceType)
        : [...prev.forceTypes, forceType];
      return { ...prev, forceTypes: types.length > 0 ? types : ['applied'] };
    });
  }, []);

  return (
    <div className="bg-gray-700/50 p-3 rounded-md">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-semibold text-gray-300 uppercase">
          Force Overlays
        </h4>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={config.enabled}
            onChange={(e) =>
              setConfig((prev) => ({ ...prev, enabled: e.target.checked }))
            }
            className="rounded border-gray-500 text-blue-500 focus:ring-blue-400"
          />
          <span className="text-xs text-gray-400">
            {config.enabled ? 'On' : 'Off'}
          </span>
        </label>
      </div>

      {config.enabled && (
        <div className="space-y-3">
          {/* Force type toggles */}
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Force Types</label>
            {FORCE_TYPE_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                className="flex items-center gap-2 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={config.forceTypes.includes(opt.value)}
                  onChange={() => toggleForceType(opt.value)}
                  className="rounded border-gray-600 text-blue-500 focus:ring-blue-400"
                />
                <span className={`text-xs ${opt.color}`}>{opt.label}</span>
              </label>
            ))}
          </div>

          {/* Scale factor slider */}
          <div>
            <label className="text-xs text-gray-400 block mb-1">
              Scale: {config.scaleFactor.toFixed(3)}
            </label>
            <input
              type="range"
              min="0.001"
              max="0.1"
              step="0.001"
              value={config.scaleFactor}
              onChange={(e) =>
                setConfig((prev) => ({
                  ...prev,
                  scaleFactor: parseFloat(e.target.value),
                }))
              }
              className="w-full h-1 bg-gray-600 rounded-lg appearance-none"
            />
          </div>

          {/* Options */}
          <div className="space-y-1">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.colorByMagnitude}
                onChange={(e) =>
                  setConfig((prev) => ({
                    ...prev,
                    colorByMagnitude: e.target.checked,
                  }))
                }
                className="rounded border-gray-600 text-blue-500 focus:ring-blue-400"
              />
              <span className="text-xs text-gray-400">Color by magnitude</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={config.showLabels}
                onChange={(e) =>
                  setConfig((prev) => ({
                    ...prev,
                    showLabels: e.target.checked,
                  }))
                }
                className="rounded border-gray-600 text-blue-500 focus:ring-blue-400"
              />
              <span className="text-xs text-gray-400">Show labels</span>
            </label>
          </div>

          {/* Summary */}
          <div className="border-t border-gray-600 pt-2 text-xs text-gray-400 space-y-1">
            <div>Total force: {totalForce.toFixed(1)} N</div>
            <div>Total torque: {totalTorque.toFixed(1)} N*m</div>
          </div>
        </div>
      )}
    </div>
  );
}
