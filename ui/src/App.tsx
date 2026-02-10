import { useState, useCallback } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SimulationPage } from './pages/Simulation';
import { DashboardPage } from './pages/Dashboard';
import { ModelExplorerPage } from './pages/ModelExplorer';
import { PuttingGreenPage } from './pages/PuttingGreen';
import { VideoAnalyzerPage } from './pages/VideoAnalyzer';
import { DataExplorerPage } from './pages/DataExplorer';
import { MotionCapturePage } from './pages/MotionCapture';
import { ToastProvider } from './components/ui/Toast';
import { DiagnosticsPanel } from './components/ui/DiagnosticsPanel';
import { HelpPanel } from './components/ui/HelpPanel';

function App() {
  const [helpOpen, setHelpOpen] = useState(false);
  const handleCloseHelp = useCallback(() => setHelpOpen(false), []);

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
        </Routes>
        <DiagnosticsPanel />
        <HelpPanel isOpen={helpOpen} onClose={handleCloseHelp} />
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
