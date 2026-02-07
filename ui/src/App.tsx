import { SimulationPage } from './pages/Simulation';
import { ToastProvider } from './components/ui/Toast';
import { DiagnosticsPanel } from './components/ui/DiagnosticsPanel';

function App() {
  return (
    <ToastProvider>
      <SimulationPage />
      <DiagnosticsPanel />
    </ToastProvider>
  );
}

export default App;
