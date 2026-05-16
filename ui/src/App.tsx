import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SimulationPage } from './pages/Simulation';
import { DashboardPage } from './pages/Dashboard';
import { ModelExplorerPage } from './pages/ModelExplorer';
import { PuttingGreenPage } from './pages/PuttingGreen';
import { VideoAnalyzerPage } from './pages/VideoAnalyzer';
import { DataExplorerPage } from './pages/DataExplorer';
import { MotionCapturePage } from './pages/MotionCapture';
import { ChatPage } from './pages/Chat';
import { TerrainPage } from './pages/Terrain';
import { DatasetGeneratorPage } from './pages/DatasetGenerator';
import { AnalysisToolsPage } from './pages/AnalysisTools';
import { ToastProvider } from './components/ui/Toast';
import { DiagnosticsPanel } from './components/ui/DiagnosticsPanel';
import { HelpPanel } from './components/ui/HelpPanel';
import { useUIStore } from './stores';

function App() {
  const helpOpen = useUIStore((s) => s.helpOpen);
  const setHelpOpen = useUIStore((s) => s.setHelpOpen);

  return (
    <BrowserRouter>
      <ToastProvider>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/simulation" element={<SimulationPage />} />
          <Route path="/tools/model-explorer" element={<ModelExplorerPage />} />
          {/* Phase 5: Tool pages (#1206) */}
          <Route path="/tools/putting-green" element={<PuttingGreenPage />} />
          <Route path="/tools/video-analyzer" element={<VideoAnalyzerPage />} />
          <Route path="/tools/data-explorer" element={<DataExplorerPage />} />
          <Route path="/tools/motion-capture" element={<MotionCapturePage />} />
          <Route path="/tools/terrain" element={<TerrainPage />} />
          <Route path="/tools/dataset" element={<DatasetGeneratorPage />} />
          <Route path="/tools/analysis" element={<AnalysisToolsPage />} />
          {/* Chat (#3505): wires chat_ws backend into the UI */}
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
        <DiagnosticsPanel />
        <HelpPanel isOpen={helpOpen} onClose={() => setHelpOpen(false)} />
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
