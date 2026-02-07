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

  it('renders without crashing', () => {
    render(<App />, { wrapper: createWrapper() });
    // At "/" the Dashboard should render
    expect(screen.getByTestId('dashboard-page-mock')).toBeInTheDocument();
  });

  it('renders DashboardPage at root route', () => {
    render(<App />, { wrapper: createWrapper() });
    expect(screen.getByText('DashboardPage Mock')).toBeInTheDocument();
  });

  it('exports default App component', () => {
    expect(App).toBeDefined();
    expect(typeof App).toBe('function');
  });
});
