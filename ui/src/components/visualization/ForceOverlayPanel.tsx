/**
 * ForceOverlayPanel - UI controls for force/torque overlay configuration (ADR-0052, #11308).
 *
 * Provides toggles for force types, body filtering, magnitude color-coding,
 * and label display. Streams real provider glyphs from the WebSocket when connected,
 * falling back to REST polling when disconnected.
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import type { ForceVector3D, ForceOverlayConfig } from './ForceOverlay';
import type { GlyphSetV1 } from '@/types/glyphs';
import { apiFetch } from '@/api/fetch';
import { usePolling } from '@/hooks/usePolling';

export interface ForceOverlayPanelProps {
  /** Primary callback when serialized GlyphSet updates (FTO-23) */
  onGlyphsChange?: (glyphs: GlyphSetV1 | null) => void;
  /** Backward-compatible callback when vector list updates */
  onVectorsChange?: (vectors: ForceVector3D[]) => void;
  /** Whether simulation is running */
  isRunning: boolean;
  /** Whether WebSocket simulation stream is currently connected */
  isConnected?: boolean;
  /** Real-time force_overlay payload from WebSocket SimulationFrame */
  socketForceOverlay?: GlyphSetV1 | Record<string, unknown> | null;
  /** Callback when configuration changes */
  onConfigChange?: (config: ForceOverlayConfig) => void;
  /** Polling interval in ms (0 to disable) */
  pollInterval?: number;
}

export const FORCE_POLL_INTERVAL_MS = 500;

const DEFAULT_CONFIG: ForceOverlayConfig = {
  enabled: false,
  forceTypes: ['applied', 'contact', 'joint_reaction'],
  scaleFactor: 0.01,
  colorByMagnitude: true,
  showLabels: false,
  bodyFilter: null,
};

const FORCE_TYPE_OPTIONS = [
  { value: 'applied', label: 'Applied Torques', color: 'text-orange-400' },
  { value: 'joint_reaction', label: 'Joint Reactions', color: 'text-blue-400' },
  { value: 'contact', label: 'Contact Forces', color: 'text-emerald-400' },
  { value: 'gravity', label: 'Gravity', color: 'text-amber-500' },
  { value: 'muscle', label: 'Muscle Forces', color: 'text-pink-400' },
  { value: 'grip', label: 'Grip Forces', color: 'text-yellow-400' },
  { value: 'external', label: 'External Forces', color: 'text-sky-400' },
];

export function ForceOverlayPanel({
  onGlyphsChange,
  onVectorsChange,
  isRunning,
  isConnected = false,
  socketForceOverlay,
  onConfigChange,
  pollInterval = FORCE_POLL_INTERVAL_MS,
}: ForceOverlayPanelProps) {
  const [config, setConfig] = useState<ForceOverlayConfig>(DEFAULT_CONFIG);
  const [polledTotals, setPolledTotals] = useState({ force: 0, torque: 0 });

  // Notify parent of config changes
  useEffect(() => {
    onConfigChange?.(config);
  }, [config, onConfigChange]);

  // When disabled, clear overlay payloads
  useEffect(() => {
    if (!config.enabled) {
      onGlyphsChange?.(null);
      onVectorsChange?.([]);
    }
  }, [config.enabled, onGlyphsChange, onVectorsChange]);

  // Handle real-time WebSocket payloads when connected
  useEffect(() => {
    if (!config.enabled || !isConnected || !socketForceOverlay) {
      return;
    }

    const glyphs = socketForceOverlay as GlyphSetV1;
    if (glyphs.schema_version === 'glyph-set-v1') {
      onGlyphsChange?.(glyphs);
    }
  }, [config.enabled, isConnected, socketForceOverlay, onGlyphsChange]);

  // Compute live totals from socket or polled REST
  const socketTotals = useMemo(() => {
    if (!config.enabled || !isConnected || !socketForceOverlay) {
      return { force: 0, torque: 0 };
    }
    const glyphs = socketForceOverlay as GlyphSetV1;
    if (glyphs.schema_version !== 'glyph-set-v1') {
      return { force: 0, torque: 0 };
    }
    let fSum = 0;
    if (Array.isArray(glyphs.arrows)) {
      for (const a of glyphs.arrows) {
        fSum += Math.abs(a.magnitude || 0);
      }
    }
    let tSum = 0;
    if (Array.isArray(glyphs.torque_arcs)) {
      for (const arc of glyphs.torque_arcs) {
        tSum += Math.abs(arc.magnitude || 0);
      }
    }
    return { force: fSum, torque: tSum };
  }, [config.enabled, isConnected, socketForceOverlay]);

  const totalForce = isConnected
    ? socketTotals.force
    : config.enabled
      ? polledTotals.force
      : 0;
  const totalTorque = isConnected
    ? socketTotals.torque
    : config.enabled
      ? polledTotals.torque
      : 0;

  // REST polling fallback when socket is disconnected
  const fetchVectors = useCallback(async () => {
    if (!config.enabled || isConnected) {
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

      const data = await apiFetch<{
        glyphs?: GlyphSetV1 | null;
        vectors?: ForceVector3D[];
        total_force_magnitude?: number;
        total_torque_magnitude?: number;
      }>(`/api/simulation/forces?${params}`);

      if (data.glyphs && data.glyphs.schema_version === 'glyph-set-v1') {
        onGlyphsChange?.(data.glyphs);
      }
      onVectorsChange?.(data.vectors || []);
      setPolledTotals({
        force: data.total_force_magnitude || 0,
        torque: data.total_torque_magnitude || 0,
      });
    } catch {
      // Silently ignore network errors during background polling
    }
  }, [config, isConnected, onGlyphsChange, onVectorsChange]);

  // Poll only while enabled, running, socket disconnected, and tab visible
  usePolling(fetchVectors, {
    intervalMs: pollInterval > 0 ? pollInterval : FORCE_POLL_INTERVAL_MS,
    enabled: config.enabled && isRunning && !isConnected && pollInterval > 0,
  });

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
          {/* Source indicator */}
          <div className="text-[10px] text-gray-400 flex items-center justify-between">
            <span>Stream Source:</span>
            <span
              className={
                isConnected
                  ? 'text-green-400 font-mono'
                  : 'text-amber-400 font-mono'
              }
            >
              {isConnected ? 'WebSocket (Real-time)' : 'REST Polling (2 Hz)'}
            </span>
          </div>

          {/* Force type toggles */}
          <div className="space-y-1">
            <label className="text-xs text-gray-400">Force Types</label>
            <div className="grid grid-cols-1 gap-1">
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
              className="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
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
