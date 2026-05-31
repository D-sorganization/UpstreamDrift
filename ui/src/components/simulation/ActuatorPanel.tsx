/**
 * ActuatorPanel - Per-actuator control sliders.
 *
 * Queries engine capabilities to dynamically generate sliders
 * for each actuator. Supports multiple control types: constant,
 * polynomial, PD gains, trajectory.
 *
 * See issue #1198
 */

import { useState, useCallback, useEffect, useMemo } from 'react';
import { PolynomialGeneratorPanel } from './PolynomialGeneratorPanel';
import { ActuatorSlider } from './ActuatorSlider';
import { apiFetch } from '@/api/fetch';

/** Actuator descriptor from the API. See issue #1198 */
export interface ActuatorInfo {
  index: number;
  name: string;
  control_type: string;
  value: number;
  min_value: number;
  max_value: number;
  units: string;
  joint_type: string;
}

/** Actuator panel state from the API. See issue #1198 */
export interface ActuatorPanelState {
  n_actuators: number;
  actuators: ActuatorInfo[];
  available_control_types: string[];
  engine_name: string;
}

interface ActuatorPanelProps {
  /** Whether the simulation is running */
  isRunning: boolean;
  /** Polling interval for state refresh (ms) */
  refreshInterval?: number;
}

/**
 * Helper to identify body regions for actuator grouping.
 */
function getActuatorGroup(name: string): string {
  const lower = name.toLowerCase();
  if (
    lower.includes('hip') ||
    lower.includes('knee') ||
    lower.includes('ankle') ||
    lower.includes('foot') ||
    lower.includes('thigh') ||
    lower.includes('leg') ||
    lower.includes('toe') ||
    lower.includes('pelvis')
  ) {
    return 'Lower Body';
  }
  if (
    lower.includes('shoulder') ||
    lower.includes('elbow') ||
    lower.includes('wrist') ||
    lower.includes('arm') ||
    lower.includes('hand') ||
    lower.includes('clavicle') ||
    lower.includes('finger')
  ) {
    return 'Upper Body';
  }
  if (
    lower.includes('spine') ||
    lower.includes('torso') ||
    lower.includes('neck') ||
    lower.includes('head') ||
    lower.includes('chest') ||
    lower.includes('back') ||
    lower.includes('waist') ||
    lower.includes('trunk') ||
    lower.includes('core')
  ) {
    return 'Core & Head';
  }
  return 'Other';
}

/**
 * ActuatorPanel provides dynamic slider controls for each actuator
 * in the active simulation engine.
 *
 * See issue #1198
 */
export function ActuatorPanel({
  isRunning,
  refreshInterval = 1000,
}: ActuatorPanelProps) {
  const [panelState, setPanelState] = useState<ActuatorPanelState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  // Grouping filter state
  const [selectedRegion, setSelectedRegion] = useState('All');
  const [collapsedRegions, setCollapsedRegions] = useState<Record<string, boolean>>({
    'Upper Body': false,
    'Lower Body': false,
    'Core & Head': false,
    'Other': false,
  });

  const fetchActuators = useCallback(async () => {
    try {
      const data = await apiFetch<ActuatorPanelState>('/api/simulation/actuators');
      setPanelState(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load actuators');
    }
  }, []);

  useEffect(() => {
    // fetchActuators is async (awaits apiFetch before any setState), so the
    // updates never run synchronously inside the effect. Scheduling via a
    // microtask makes that explicit for react-hooks/set-state-in-effect.
    const refresh = () => void Promise.resolve().then(fetchActuators);
    refresh();

    if (isRunning && refreshInterval > 0) {
      const interval = setInterval(refresh, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchActuators, isRunning, refreshInterval]);

  const handleValueChange = useCallback(async (index: number, value: number) => {
    // #6898: surface failures (bad index, engine not loaded) via setError
    // instead of silently swallowing them, which left the slider moved while
    // the engine was unchanged.
    try {
      await apiFetch('/api/simulation/actuators', {
        method: 'POST',
        body: JSON.stringify({
          actuator_index: index,
          value,
          control_type: 'constant',
        }),
      });
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Failed to update actuator value',
      );
    }
  }, []);

  const handleControlTypeChange = useCallback(
    async (index: number, controlType: string) => {
      try {
        await apiFetch('/api/simulation/actuators', {
          method: 'POST',
          body: JSON.stringify({
            actuator_index: index,
            value: 0,
            control_type: controlType,
          }),
        });
        setError(null);
        fetchActuators();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to change control type',
        );
      }
    },
    [fetchActuators],
  );

  const handleApplyPolynomial = useCallback(
    async (index: number, coefficients: number[]) => {
      try {
        await apiFetch('/api/simulation/actuators', {
          method: 'POST',
          body: JSON.stringify({
            actuator_index: index,
            value: 0,
            control_type: 'polynomial',
            parameters: { coefficients },
          }),
        });
        setError(null);
        fetchActuators();
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to apply polynomial',
        );
      }
    },
    [fetchActuators],
  );

  // Group logic
  const shouldGroup = panelState && panelState.actuators.length > 20;

  const groupedActuators = useMemo(() => {
    if (!panelState) return {};
    const groups: Record<string, ActuatorInfo[]> = {};
    for (const act of panelState.actuators) {
      const groupName = shouldGroup ? getActuatorGroup(act.name) : 'All';
      if (!groups[groupName]) {
        groups[groupName] = [];
      }
      groups[groupName].push(act);
    }
    return groups;
  }, [panelState, shouldGroup]);

  const activeRegions = useMemo(() => {
    if (!shouldGroup) return ['All'];
    if (selectedRegion !== 'All') return [selectedRegion];
    return ['Upper Body', 'Lower Body', 'Core & Head', 'Other'];
  }, [shouldGroup, selectedRegion]);

  const renderedRegions = useMemo(() => {
    return activeRegions.filter(
      (r) => groupedActuators[r] && groupedActuators[r].length > 0
    );
  }, [activeRegions, groupedActuators]);

  return (
    <div className="bg-gray-700/50 p-3 rounded-md space-y-3">
      {/* Header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex items-center justify-between w-full text-left"
      >
        <h4 className="text-xs font-semibold text-gray-300 uppercase">
          Actuator Controls
        </h4>
        <span className="text-xs text-gray-500">
          {collapsed ? '+' : '-'}
          {panelState ? ` (${panelState.n_actuators})` : ''}
        </span>
      </button>

      {!collapsed && (
        <div className="space-y-3">
          {error && (
            <div className="text-xs text-red-400 bg-red-900/20 p-2 rounded">
              {error}
            </div>
          )}

          {panelState && panelState.actuators.length > 0 ? (
            <>
              <div className="text-xs text-gray-500">
                Engine: {panelState.engine_name}
              </div>

              {/* Polynomial Panel */}
              <PolynomialGeneratorPanel
                actuators={panelState.actuators}
                onApplyPolynomial={handleApplyPolynomial}
              />

              {/* Region Filter Selector */}
              {shouldGroup && (
                <div className="flex flex-col gap-1">
                  <label htmlFor="region-filter-select" className="text-xs text-gray-400">
                    Filter by region
                  </label>
                  <select
                    id="region-filter-select"
                    value={selectedRegion}
                    onChange={(e) => setSelectedRegion(e.target.value)}
                    className="text-xs bg-gray-600 text-gray-300 rounded px-2 py-1.5 border-none focus:ring-1 focus:ring-blue-400"
                  >
                    <option value="All">All Regions</option>
                    <option value="Upper Body">Upper Body</option>
                    <option value="Lower Body">Lower Body</option>
                    <option value="Core & Head">Core & Head</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              )}

              {/* Actuators List */}
              <div className="space-y-2">
                {renderedRegions.map((regionName) => {
                  const acts = groupedActuators[regionName] || [];
                  const isCollapsed = collapsedRegions[regionName];

                  if (regionName === 'All') {
                    return acts.map((act) => (
                      <ActuatorSlider
                        key={act.index}
                        actuator={act}
                        onValueChange={handleValueChange}
                        onControlTypeChange={handleControlTypeChange}
                        availableTypes={panelState.available_control_types}
                      />
                    ));
                  }

                  return (
                    <div key={regionName} className="space-y-1">
                      <button
                        type="button"
                        onClick={() =>
                          setCollapsedRegions((prev) => ({
                            ...prev,
                            [regionName]: !prev[regionName],
                          }))
                        }
                        className="w-full flex items-center justify-between text-xs font-semibold bg-gray-800/40 hover:bg-gray-800/60 p-2 rounded transition-colors"
                      >
                        <span className="text-gray-300">
                          {regionName} ({acts.length})
                        </span>
                        <span className="text-gray-500">{isCollapsed ? '+' : '-'}</span>
                      </button>

                      {!isCollapsed && (
                        <div className="space-y-2 pl-1 pt-1">
                          {acts.map((act) => (
                            <ActuatorSlider
                              key={act.index}
                              actuator={act}
                              onValueChange={handleValueChange}
                              onControlTypeChange={handleControlTypeChange}
                              availableTypes={panelState.available_control_types}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="text-xs text-gray-500 italic text-center py-2">
              {panelState ? 'No actuators available' : 'Loading actuators...'}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
