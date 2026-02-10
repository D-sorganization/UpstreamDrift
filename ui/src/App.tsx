import { useState, useCallback } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SimulationPage } from './pages/Simulation';
import { DashboardPage } from './pages/Dashboard';
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
        </Routes>
        <DiagnosticsPanel />
        <HelpPanel isOpen={helpOpen} onClose={handleCloseHelp} />
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
