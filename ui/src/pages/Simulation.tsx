import { useState, useCallback, useEffect } from 'react';
import { useSimulation } from '@/api/client';
import { EngineSelector } from '@/components/simulation/EngineSelector';
import { SimulationControls } from '@/components/simulation/SimulationControls';
import { ParameterPanel, type SimulationParameters } from '@/components/simulation/ParameterPanel';
import { Scene3D } from '@/components/visualization/Scene3D';
import { LivePlot } from '@/components/analysis/LivePlot';
import { ConnectionStatus } from '@/components/ui/ConnectionStatus';
import { useToast } from '@/components/ui/Toast';

export function SimulationPage() {
  const [selectedEngine, setSelectedEngine] = useState<string>('mujoco');
  const [parameters, setParameters] = useState<SimulationParameters>({
    duration: 3.0,
    timestep: 0.002,
    liveAnalysis: true,
    gpuAcceleration: false,
  });
  const { showSuccess, showError, showInfo } = useToast();

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
  } = useSimulation(selectedEngine);

  // Handle parameter changes
  const handleParameterChange = useCallback((params: SimulationParameters) => {
    setParameters(params);
  }, []);

  // Start simulation with current parameters
  const handleStart = useCallback(() => {
    start({
      duration: parameters.duration,
      timestep: parameters.timestep,
      live_analysis: parameters.liveAnalysis,
    });
    showInfo(`Starting ${selectedEngine} simulation...`);
  }, [start, parameters, selectedEngine, showInfo]);

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

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Sidebar - Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto flex-shrink-0 z-10">
        <h2 className="text-xl font-bold text-white mb-6">Golf Suite</h2>

        {/* Connection Status */}
        <div className="mb-4">
          <ConnectionStatus status={connectionStatus} />
        </div>

        <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Physics Engine</h3>
            <EngineSelector
            value={selectedEngine}
            onChange={setSelectedEngine}
            disabled={isRunning}
            />
        </div>

        {/* Parameter Panel */}
        <div className="mb-6 border-t border-gray-700 pt-4">
          <ParameterPanel
            engine={selectedEngine}
            disabled={isRunning}
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
            />
        </div>
      </aside>

      {/* Main Content - 3D View + Analysis */}
      <main className="flex-1 flex flex-col relative min-w-0">
        {/* 3D Visualization */}
        <div className="flex-1 relative bg-gray-950">
          <Scene3D
            engine={selectedEngine}
            frame={currentFrame}
            frames={frames}
          />

          {/* Overlay: Status */}
          <div className="absolute top-4 left-4 bg-black/70 backdrop-blur-sm px-4 py-2 rounded-lg border border-white/10 shadow-lg pointer-events-none">
            <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`} />
                <span className="text-gray-200 font-mono text-sm">
                {isRunning ? `Frame ${currentFrame?.frame || 0}` : 'Ready'}
                </span>
            </div>
            {currentFrame && (
                <div className="mt-1 text-xs text-gray-400 font-mono">
                    Time: {currentFrame.time.toFixed(3)}s
                </div>
            )}
          </div>
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
                 Start simulation to view live data
             </div>
        )}
      </aside>
    </div>
  );
}
