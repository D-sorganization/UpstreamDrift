import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { getBackendStatus, isTauri } from './api/backend';
import App from './App';
import './index.css';

const queryClient = new QueryClient();

async function bootstrap(): Promise<void> {
  if (isTauri()) {
    try {
      await getBackendStatus();
    } catch {
      // Diagnostics panel surfaces backend start failures.
    }
  }

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <QueryClientProvider client={queryClient}>
        <ErrorBoundary>
          <App />
        </ErrorBoundary>
      </QueryClientProvider>
    </React.StrictMode>,
  );
}

void bootstrap();
