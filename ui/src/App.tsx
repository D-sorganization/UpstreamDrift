import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SimulationPage } from './pages/Simulation';
import { DashboardPage } from './pages/Dashboard';
import { ToastProvider } from './components/ui/Toast';
import { DiagnosticsPanel } from './components/ui/DiagnosticsPanel';

function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/simulation" element={<SimulationPage />} />
        </Routes>
        <DiagnosticsPanel />
      </ToastProvider>
    </BrowserRouter>
  );
}

export default App;
