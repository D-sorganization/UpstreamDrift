import { useState } from 'react';
import { useSimulation } from '@/api/client';
import { EngineSelector } from '@/components/simulation/EngineSelector';
import { SimulationControls } from '@/components/simulation/SimulationControls';
import { Scene3D } from '@/components/visualization/Scene3D';
// import { LivePlot } from '../components/analysis/LivePlot'; // To be implemented

export function SimulationPage() {
  const [selectedEngine, setSelectedEngine] = useState<string>('mujoco');
  const {
    isRunning,
    isPaused,
    currentFrame,
    start,
    stop,
    pause,
    resume
  } = useSimulation(selectedEngine);

  return (
    <div className="flex h-screen bg-gray-900 overflow-hidden">
      {/* Left Sidebar - Controls */}
      <aside className="w-80 bg-gray-800 border-r border-gray-700 p-4 overflow-y-auto flex-shrink-0 z-10">
        <h2 className="text-xl font-bold text-white mb-6">Golf Suite</h2>

        <div className="mb-6">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Simulation</h3>
            <EngineSelector
            value={selectedEngine}
            onChange={setSelectedEngine}
            disabled={isRunning}
            />
        </div>

        {/* <ParameterPanel engine={selectedEngine} /> */}

        <div className="mt-auto pt-6 border-t border-gray-700">
            <SimulationControls
            isRunning={isRunning}
            isPaused={isPaused}
            onStart={() => start()} // Can pass config here
            onStop={stop}
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

        {/* Bottom: Live Analysis (Placeholder) */}
        {/* 
        <div className="h-64 bg-gray-800 border-t border-gray-700">
          <LivePlot data={currentFrame?.analysis} />
        </div> 
        */}
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
