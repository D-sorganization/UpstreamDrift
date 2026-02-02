import { SimulationPage } from './pages/Simulation';
import { ToastProvider } from './components/ui/Toast';

function App() {
  return (
    <ToastProvider>
      <SimulationPage />
    </ToastProvider>
  );
}

export default App;
