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
import { CharacterBuilderPage } from './pages/CharacterBuilder';
import { CanonicalCoreShellPage } from './pages/CanonicalCoreShell';
import { NotFoundPage } from './pages/NotFound';
import { ScrollToTop } from './utils/ScrollToTop';
import { RouteTitle } from './utils/RouteTitle';
import { BallFlightPage } from './pages/BallFlight';
import { SettingsPage } from './pages/Settings';
import { ToastProvider } from './components/ui/Toast';
import { useWebSettingsBootstrap } from './api/useWebSettings';
import { DiagnosticsPanel } from './components/ui/DiagnosticsPanel';
import { HelpPanel } from './components/ui/HelpPanel';
import { useUIStore } from './stores';

function App() {
  const helpOpen = useUIStore((s) => s.helpOpen);
  const setHelpOpen = useUIStore((s) => s.setHelpOpen);

  // Load persisted web settings (font scale, simulation defaults) at app
  // start; server file is the source of truth, localStorage is a cache (#7457).
  useWebSettingsBootstrap();

  return (
    <BrowserRouter>
      <ScrollToTop />
      <RouteTitle />
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
          <Route path="/tools/character-builder" element={<CharacterBuilderPage />} />
          <Route
            path="/tools/canonical-core/estimation"
            element={<CanonicalCoreShellPage mode="estimation" />}
          />
          <Route
            path="/tools/canonical-core/comparison"
            element={<CanonicalCoreShellPage mode="comparison" />}
          />
          {/* Shot Tracer / ball-flight comparison (#7456) */}
          <Route path="/ball-flight" element={<BallFlightPage />} />
          {/* Chat (#3505): wires chat_ws backend into the UI */}
          <Route path="/chat" element={<ChatPage />} />
          {/* Settings (#7457): server-persisted preferences surface */}
          <Route path="/settings" element={<SettingsPage />} />
          {/* Catch-all 404 (#7430) — must stay last. */}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
        <DiagnosticsPanel />
        <HelpPanel isOpen={helpOpen} onClose={() => setHelpOpen(false)} />
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
