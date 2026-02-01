import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen } from '@testing-library/dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the SimulationPage to isolate App component testing
vi.mock('@/pages/Simulation', () => ({
  SimulationPage: () => <div data-testid="simulation-page-mock">SimulationPage Mock</div>,
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

  it('renders without crashing', () => {
    render(<App />, { wrapper: createWrapper() });

    expect(screen.getByTestId('simulation-page-mock')).toBeInTheDocument();
  });

  it('renders SimulationPage component', () => {
    render(<App />, { wrapper: createWrapper() });

    expect(screen.getByText('SimulationPage Mock')).toBeInTheDocument();
  });

  it('exports default App component', () => {
    expect(App).toBeDefined();
    expect(typeof App).toBe('function');
  });
});
