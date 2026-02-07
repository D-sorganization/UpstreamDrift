import { useState, useCallback, useEffect } from 'react';
import { useSimulation } from '@/api/client';
import { useEngineManager } from '@/api/useEngineManager';
import { EngineSelector } from '@/components/simulation/EngineSelector';
import { SimulationControls } from '@/components/simulation/SimulationControls';
import { ParameterPanel, type SimulationParameters } from '@/components/simulation/ParameterPanel';
import { Scene3D } from '@/components/visualization/Scene3D';
import { LivePlot } from '@/components/analysis/LivePlot';
import { ConnectionStatus } from '@/components/ui/ConnectionStatus';
import { useToast } from '@/components/ui/Toast';

export function SimulationPage() {
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null);
  const [parameters, setParameters] = useState<SimulationParameters>({
    duration: 3.0,
    timestep: 0.002,
    liveAnalysis: true,
    gpuAcceleration: false,
  });
  const { showSuccess, showError, showInfo } = useToast();

  // Lazy engine manager — no engines loaded on startup
  const { engines, loadedEngines, requestLoad, unloadEngine } = useEngineManager();

  // Only connect to the simulation when an engine is selected AND loaded
  const activeEngine = selectedEngine || 'mujoco'; // fallback for hook (won't connect unless started)
  const {
    isRunning,
    isPaused,
    currentFrame,
    frames,
    connectionStatus,
    start,
    stop,
    pause,
    resume
  } = useSimulation(activeEngine);

  // Auto-select first loaded engine if none selected
  useEffect(() => {
    if (!selectedEngine && loadedEngines.length > 0) {
      setSelectedEngine(loadedEngines[0].name);
    }
  }, [selectedEngine, loadedEngines]);

  // Handle engine loading
  const handleLoadEngine = useCallback(async (engineName: string) => {
    showInfo(`Loading ${engineName}...`);
    await requestLoad(engineName);
    showSuccess(`${engineName} engine loaded`);
  }, [requestLoad, showInfo, showSuccess]);

  // Handle engine unloading
  const handleUnloadEngine = useCallback((engineName: string) => {
    if (selectedEngine === engineName) {
      // Deselect if unloading the active engine
      const remaining = loadedEngines.filter((e) => e.name !== engineName);
      setSelectedEngine(remaining.length > 0 ? remaining[0].name : null);
    }
    unloadEngine(engineName);
    showInfo(`${engineName} engine unloaded`);
  }, [selectedEngine, loadedEngines, unloadEngine, showInfo]);

  // Handle parameter changes
  const handleParameterChange = useCallback((params: SimulationParameters) => {
    setParameters(params);
  }, []);

  // Start simulation with current parameters
  const handleStart = useCallback(() => {
    if (!selectedEngine) {
      showError('Please load and select an engine first');
      return;
    }
    start({
      duration: parameters.duration,
      timestep: parameters.timestep,
      live_analysis: parameters.liveAnalysis,
    });
    showInfo(`Starting ${selectedEngine} simulation...`);
  }, [start, parameters, selectedEngine, showInfo, showError]);

  // Handle stop
  const handleStop = useCallback(() => {
    stop();
    showInfo('Simulation stopped');
  }, [stop, showInfo]);

  // Show toast on connection status changes
  useEffect(() => {
    if (connectionStatus === 'connected') {
      showSuccess('Connected to simulation server');
    } else if (connectionStatus === 'failed') {
      showError('Connection failed. Please check the server.');
    }
  }, [connectionStatus, showSuccess, showError]);

  const canStart = selectedEngine !== null && !isRunning;

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
            selectedEngine={selectedEngine}
            onSelect={setSelectedEngine}
            onLoad={handleLoadEngine}
            onUnload={handleUnloadEngine}
            disabled={isRunning}
          />
        </div>

        {/* Parameter Panel */}
        <div className="mb-6 border-t border-gray-700 pt-4">
          <ParameterPanel
            engine={selectedEngine || 'mujoco'}
            disabled={isRunning || !selectedEngine}
            onChange={handleParameterChange}
          />
        </div>

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
            engine={selectedEngine || 'mujoco'}
            frame={currentFrame}
            frames={frames}
          />

          {/* Overlay: Status */}
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-4 py-2 rounded-lg border border-white/10 shadow-lg pointer-events-none">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
              <span className="text-gray-200 font-mono text-sm">
                {isRunning ? `Frame ${currentFrame?.frame || 0}` : selectedEngine ? 'Ready' : 'No engine loaded'}
              </span>
            </div>
            {currentFrame && (
              <div className="mt-1 text-xs text-gray-400 font-mono">
                Time: {currentFrame.time.toFixed(3)}s
              </div>
            )}
          </div>

          {/* No engine loaded overlay */}
          {!selectedEngine && !isRunning && (
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
      <aside className="w-72 bg-gray-800 border-l border-gray-700 p-4 flex-shrink-0 z-10">
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
            {selectedEngine
              ? 'Start simulation to view live data'
              : 'Load an engine to get started'}
          </div>
        )}
      </aside>
    </div>
  );
}
