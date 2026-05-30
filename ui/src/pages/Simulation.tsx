import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { useSimulation } from '@/api/client';
import type { ConnectionStatus as ConnectionStatusType } from '@/api/client';
import { useEngineStore, selectEffectiveEngine } from '@/stores/useEngineStore';
import { useSimulationStore } from '@/stores/useSimulationStore';
import { EngineSelector } from '@/components/simulation/EngineSelector';
import { SimulationControls } from '@/components/simulation/SimulationControls';
import { ParameterPanel, type SimulationParameters } from '@/components/simulation/ParameterPanel';
import { ActuatorPanel } from '@/components/simulation/ActuatorPanel';
import { Scene3D } from '@/components/visualization/Scene3D';
import { ForceOverlayPanel } from '@/components/visualization/ForceOverlayPanel';
import type { ForceVector3D } from '@/components/visualization/ForceOverlay';
import { LivePlot } from '@/components/analysis/LivePlot';
import { ConnectionStatus } from '@/components/ui/ConnectionStatus';
import { useToast } from '@/components/ui/Toast';

export function SimulationPage() {
  // ── Global stores ─────────────────────────────────────────────────────
  const engines = useEngineStore((s) => s.engines);
  const selectEngine = useEngineStore((s) => s.selectEngine);
  const requestLoad = useEngineStore((s) => s.requestLoad);
  const unloadEngine = useEngineStore((s) => s.unloadEngine);
  // Uses the canonical selector from the store (F8: removes duplicated derivation)
  const effectiveEngine = useEngineStore(selectEffectiveEngine);

  const loadedEngines = useMemo(
    () => engines.filter((e) => e.loadState === 'loaded'),
    [engines]
  );

  const parameters = useSimulationStore((s) => s.parameters);
  const replaceParameters = useSimulationStore((s) => s.replaceParameters);
  const markRun = useSimulationStore((s) => s.markRun);

  // ── Local state (component-specific) ──────────────────────────────────
  const { showSuccess, showError, showInfo } = useToast();
  const [forceVectors, setForceVectors] = useState<ForceVector3D[]>([]);
  const [speedFactor, setSpeedFactor] = useState<number>(1.0);

  // Only connect to the simulation when an engine is selected AND loaded
  const activeEngine = effectiveEngine || 'mujoco';
  const {
    isRunning,
    isPaused,
    currentFrame,
    frames,
    connectionStatus,
    wsError,
    start,
    stop,
    pause,
    resume,
    setSpeed,
  } = useSimulation(activeEngine);

  const handleSpeedChange = useCallback(async (value: number) => {
    setSpeedFactor(value);
    try {
      await setSpeed(value);
    } catch (err) {
      showError(err instanceof Error ? err.message : 'Failed to update speed');
    }
  }, [setSpeed, showError]);

  // ── Event handlers ────────────────────────────────────────────────────

  const handleLoadEngine = useCallback(
    async (engineName: string) => {
      showInfo(`Loading ${engineName}...`);
      try {
        await requestLoad(engineName);
        showSuccess(`${engineName} engine loaded`);
      } catch (err) {
        const message = err instanceof Error ? err.message : `Failed to load ${engineName}`;
        showError(message);
      }
    },
    [requestLoad, showInfo, showSuccess, showError]
  );

  const handleUnloadEngine = useCallback(
    (engineName: string) => {
      unloadEngine(engineName);
      showInfo(`${engineName} engine unloaded`);
    },
    [unloadEngine, showInfo]
  );

  const handleParameterChange = useCallback(
    (params: SimulationParameters) => {
      replaceParameters(params);
    },
    [replaceParameters]
  );

  const handleStart = useCallback(() => {
    if (!effectiveEngine) {
      showError('Please load and select an engine first');
      return;
    }
    setSpeedFactor(1.0);
    start({
      duration: parameters.duration,
      timestep: parameters.timestep,
      live_analysis: parameters.liveAnalysis,
    });
    markRun();
    showInfo(`Starting ${effectiveEngine} simulation...`);
  }, [start, parameters, effectiveEngine, showInfo, showError, markRun]);

  const handleStop = useCallback(() => {
    stop();
    showInfo('Simulation stopped');
  }, [stop, showInfo]);

  // Show toast only when connection status transitions (F9: prevents spam)
  const prevConnectionStatusRef = useRef<ConnectionStatusType | null>(null);
  useEffect(() => {
    if (connectionStatus === prevConnectionStatusRef.current) return;
    prevConnectionStatusRef.current = connectionStatus;
    if (connectionStatus === 'connected') {
      showSuccess('Connected to simulation server');
    } else if (connectionStatus === 'reconnecting') {
      // F5: a transient drop re-runs connect() which restarts the sim from
      // t=0; warn the user so a silent restart is not mistaken for a freeze.
      showInfo('Connection lost — reconnecting (simulation will restart)');
    } else if (connectionStatus === 'failed') {
      showError('Connection failed. Please check the server.');
    }
  }, [connectionStatus, showSuccess, showError, showInfo]);

  // Surface WebSocket errors via toast (F2)
  useEffect(() => {
    if (wsError) {
      showError(wsError);
    }
  }, [wsError, showError]);

  // See issue #1199: Convert force vectors to Scene3D overlay format
  const sceneForceOverlays = useMemo(() => {
    return forceVectors.map((v) => ({
      origin: v.origin as [number, number, number],
      direction: v.direction as [number, number, number],
      magnitude: v.magnitude,
      color: `rgb(${Math.round(v.color[0] * 255)}, ${Math.round(v.color[1] * 255)}, ${Math.round(v.color[2] * 255)})`,
      label: v.label ?? undefined,
    }));
  }, [forceVectors]);

  const canStart = effectiveEngine !== null && !isRunning;

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Sidebar - Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto flex-shrink-0 z-10">
        <h2 className="text-xl font-bold text-white mb-2">Golf Suite</h2>
        <p className="text-xs text-gray-500 mb-6">
          {loadedEngines.length} engine{loadedEngines.length !== 1 ? 's' : ''} loaded
        </p>

        {/* Connection Status */}
        <div className="mb-4">
          <ConnectionStatus status={connectionStatus} />
        </div>

        {/* Engine Selector with lazy loading */}
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Physics Engines</h3>
          <EngineSelector
            engines={engines}
            selectedEngine={effectiveEngine}
            onSelect={selectEngine}
            onLoad={handleLoadEngine}
            onUnload={handleUnloadEngine}
            disabled={isRunning}
          />
        </div>

        {/* Parameter Panel */}
        <div className="mb-6 border-t border-gray-700 pt-4">
          <ParameterPanel
            engine={effectiveEngine || 'mujoco'}
            disabled={isRunning || !effectiveEngine}
            onChange={handleParameterChange}
          />
        </div>

        {/* See issue #1198: Per-actuator controls */}
        <div className="mb-6 border-t border-gray-700 pt-4">
          <ActuatorPanel isRunning={isRunning} />
        </div>

        {/* Speed Factor Control */}
        {isRunning && (
          <div className="mb-4 border-t border-gray-700 pt-4">
            <label htmlFor="speed-factor" className="block text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
              Speed Factor ({speedFactor.toFixed(1)}x)
            </label>
            <input
              id="speed-factor"
              type="range"
              min="0.1"
              max="5.0"
              step="0.1"
              value={speedFactor}
              onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <div className="flex justify-between text-[10px] text-gray-500 mt-1">
              <span>0.1x</span>
              <span>1.0x (Real-time)</span>
              <span>5.0x</span>
            </div>
          </div>
        )}

        <div className="mt-auto pt-6 border-t border-gray-700">
          <SimulationControls
            isRunning={isRunning}
            isPaused={isPaused}
            onStart={handleStart}
            onStop={handleStop}
            onPause={pause}
            onResume={resume}
            disabled={!canStart && !isRunning}
          />
        </div>
      </aside>

      {/* Main Content - 3D View + Analysis */}
      <main className="flex-1 flex flex-col relative min-w-0">
        {/* 3D Visualization */}
        <div className="flex-1 relative bg-gray-950">
          <Scene3D
            engine={effectiveEngine || 'mujoco'}
            frame={currentFrame}
            frames={frames}
            forceOverlays={sceneForceOverlays}
          />

          {/* Overlay: Status */}
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-4 py-2 rounded-lg border border-white/10 shadow-lg pointer-events-none">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
              <span className="text-gray-200 font-mono text-sm">
                {isRunning ? `Frame ${currentFrame?.frame || 0}` : effectiveEngine ? 'Ready' : 'No engine loaded'}
              </span>
            </div>
            {currentFrame && (
              <div className="mt-1 text-xs text-gray-400 font-mono">
                Time: {currentFrame.time.toFixed(3)}s
              </div>
            )}
          </div>

          {/* No engine loaded overlay */}
          {!effectiveEngine && !isRunning && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center">
                <div className="text-4xl mb-4 text-gray-600">⚡</div>
                <h3 className="text-lg font-semibold text-gray-400 mb-2">Load a Physics Engine</h3>
                <p className="text-sm text-gray-500 max-w-xs">
                  Select and load an engine from the sidebar to begin simulation.
                  Only loaded engines consume system resources.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Bottom: Live Analysis Charts */}
        <div className="h-64 bg-gray-800 border-t border-gray-700 p-2">
          <LivePlot frames={frames} maxPoints={200} />
        </div>
      </main>

      {/* Right Sidebar - Live Data */}
      <aside className="w-72 bg-gray-800 border-l border-gray-700 p-4 flex-shrink-0 z-10 overflow-y-auto">
        {/* See issue #1199: Force overlay controls */}
        <div className="mb-4">
          <ForceOverlayPanel
            onVectorsChange={setForceVectors}
            isRunning={isRunning}
          />
        </div>

        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Live Analysis</h3>

        {currentFrame ? (
          <div className="space-y-4">
            <div className="bg-gray-700/50 p-3 rounded-md">
              <h4 className="text-xs text-gray-400 mb-2">Simulation State</h4>
              <pre className="text-xs text-green-400 font-mono overflow-auto max-h-40">
                {JSON.stringify(currentFrame.state, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <div className="text-sm text-gray-500 italic text-center mt-10">
            {effectiveEngine
              ? 'Start simulation to view live data'
              : 'Load an engine to get started'}
          </div>
        )}
      </aside>
    </div>
  );
}
