import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { EngineSelector } from './EngineSelector';

// Mock the API client
vi.mock('@/api/client', () => ({
  fetchEngines: vi.fn(),
}));

import { fetchEngines } from '@/api/client';

const mockEngines = [
  { name: 'mujoco', available: true, loaded: true },
  { name: 'drake', available: true, loaded: false },
  { name: 'pinocchio', available: false, loaded: false },
];

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe('EngineSelector', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('loading state', () => {
    it('shows loading indicator while fetching engines', () => {
      vi.mocked(fetchEngines).mockReturnValue(new Promise(() => {})); // Never resolves

      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      expect(screen.getByRole('status', { name: /loading/i })).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when fetch fails', async () => {
      vi.mocked(fetchEngines).mockRejectedValue(new Error('Network error'));

      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(screen.getByRole('alert')).toBeInTheDocument();
      });

      expect(screen.getByText(/failed to load engines/i)).toBeInTheDocument();
    });

    it('shows retry button on error', async () => {
      vi.mocked(fetchEngines).mockRejectedValue(new Error('Network error'));

      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
      });
    });
  });

  describe('success state', () => {
    beforeEach(() => {
      vi.mocked(fetchEngines).mockResolvedValue(mockEngines);
    });

    it('renders list of engines', async () => {
      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(screen.getByText('mujoco')).toBeInTheDocument();
        expect(screen.getByText('drake')).toBeInTheDocument();
        expect(screen.getByText('pinocchio')).toBeInTheDocument();
      });
    });

    it('calls onChange when engine is selected', async () => {
      const onChange = vi.fn();

      render(
        <EngineSelector value="mujoco" onChange={onChange} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(screen.getByText('drake')).toBeInTheDocument();
      });

      fireEvent.click(screen.getByRole('radio', { name: /drake/i }));
      expect(onChange).toHaveBeenCalledWith('drake');
    });

    it('disables unavailable engines', async () => {
      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        const pinocchioButton = screen.getByRole('radio', { name: /pinocchio.*not installed/i });
        expect(pinocchioButton).toBeDisabled();
      });
    });

    it('shows selected engine with aria-checked', async () => {
      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        const selectedRadio = screen.getByRole('radio', { name: /mujoco/i });
        expect(selectedRadio).toHaveAttribute('aria-checked', 'true');
      });
    });
  });

  describe('accessibility', () => {
    beforeEach(() => {
      vi.mocked(fetchEngines).mockResolvedValue(mockEngines);
    });

    it('has radiogroup role with label', async () => {
      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(screen.getByRole('radiogroup', { name: /physics engine/i })).toBeInTheDocument();
      });
    });

    it('buttons have focus rings', async () => {
      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        const radio = screen.getByRole('radio', { name: /mujoco/i });
        expect(radio.className).toContain('focus:ring');
      });
    });
  });

  describe('disabled state', () => {
    beforeEach(() => {
      vi.mocked(fetchEngines).mockResolvedValue(mockEngines);
    });

    it('disables all engines when disabled prop is true', async () => {
      render(
        <EngineSelector value="mujoco" onChange={vi.fn()} disabled />,
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        const radios = screen.getAllByRole('radio');
        radios.forEach(radio => {
          expect(radio).toBeDisabled();
        });
      });
    });
  });
});
