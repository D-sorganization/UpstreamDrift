import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SimulationPage } from './pages/Simulation';
import { DashboardPage } from './pages/Dashboard';
import { ModelExplorerPage } from './pages/ModelExplorer';
import { DataExplorerPage } from './pages/DataExplorer';
import { ToastProvider } from './components/ui/Toast';
import { DiagnosticsPanel } from './components/ui/DiagnosticsPanel';
import { HelpPanel } from './components/ui/HelpPanel';
import { ChatPanel } from './components/ai/ChatPanel';
import { ExpertiseLevelDropdown } from './components/ui/ExpertiseLevelDropdown';
import { useUIStore } from './stores';

function App() {
  const helpOpen = useUIStore((s) => s.helpOpen);
  const helpTopicId = useUIStore((s) => s.helpTopicId);
  const setHelpOpen = useUIStore((s) => s.setHelpOpen);

  return (
    <BrowserRouter>
      <ToastProvider>
        {/* Thin global header (#3165) */}
        <header className="fixed top-0 right-0 z-40 flex items-center gap-2 p-2">
          <ExpertiseLevelDropdown />
        </header>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/simulation" element={<SimulationPage />} />
          <Route path="/tools/model-explorer" element={<ModelExplorerPage />} />
          <Route path="/tools/data-explorer" element={<DataExplorerPage />} />
          {/* WIP: putting-green, video-analyzer, motion-capture routes removed until
              backends are implemented. Tracked in #3166. */}
        </Routes>
        <DiagnosticsPanel />
        <HelpPanel
          isOpen={helpOpen}
          initialTopicId={helpTopicId ?? undefined}
          onClose={() => setHelpOpen(false)}
        />
        <ChatPanel />
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
