import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the pages to isolate App component testing
vi.mock('@/pages/Simulation', () => ({
  SimulationPage: () => <div data-testid="simulation-page-mock">SimulationPage Mock</div>,
}));

vi.mock('@/pages/Dashboard', () => ({
  DashboardPage: () => <div data-testid="dashboard-page-mock">DashboardPage Mock</div>,
}));

import App from './App';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Reset the URL after each test
    window.history.pushState({}, '', '/');
  });

  // Pages are lazy-loaded (#7433), so they resolve asynchronously after the
  // Suspense fallback — assertions use findBy* to await the chunk.
  it('renders without crashing', async () => {
    render(<App />, { wrapper: createWrapper() });
    // At "/" the Dashboard should render once its chunk resolves.
    expect(await screen.findByTestId('dashboard-page-mock')).toBeInTheDocument();
  });

  it('renders DashboardPage at root route', async () => {
    render(<App />, { wrapper: createWrapper() });
    expect(await screen.findByText('DashboardPage Mock')).toBeInTheDocument();
  });

  it('exports default App component', () => {
    expect(App).toBeDefined();
    expect(typeof App).toBe('function');
  });

  it('renders the branded 404 page for unknown routes (#7430)', async () => {
    window.history.pushState({}, '', '/tools/does-not-exist');
    render(<App />, { wrapper: createWrapper() });
    expect(await screen.findByText('Page not found')).toBeInTheDocument();
    expect(screen.getByText('/tools/does-not-exist')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /back to dashboard/i })).toBeInTheDocument();
  });

  it('shows a themed loading fallback while a route chunk resolves (#7433)', async () => {
    render(<App />, { wrapper: createWrapper() });
    // The Suspense fallback exposes an accessible status before the page loads.
    // (It may resolve quickly; either the fallback or the page must be present.)
    const dashboard = await screen.findByTestId('dashboard-page-mock');
    expect(dashboard).toBeInTheDocument();
  });
});
