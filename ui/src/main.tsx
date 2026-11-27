import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import App from './App';
import { ensureLauncherCapabilityToken } from './api/websocketToken';
import './index.css';

const queryClient = new QueryClient();

// Read the launcher capability early so the first API requests carry it.
void ensureLauncherCapabilityToken();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </QueryClientProvider>
  </React.StrictMode>,
);
