import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import { screen, fireEvent } from '@testing-library/dom';
import { EngineSelector } from './EngineSelector';
import type { ManagedEngine } from '@/api/useEngineManager';

const createEngine = (overrides: Partial<ManagedEngine> = {}): ManagedEngine => ({
  name: 'mujoco',
  displayName: 'MuJoCo',
  description: 'High-performance physics',
  loadState: 'idle',
  available: true,
  capabilities: ['rigid_body'],
  ...overrides,
});

const mockEngines: ManagedEngine[] = [
  createEngine({ name: 'mujoco', displayName: 'MuJoCo', description: 'High-performance physics for robotics', loadState: 'loaded' }),
  createEngine({ name: 'drake', displayName: 'Drake', description: 'Optimization-based dynamics', loadState: 'idle' }),
  createEngine({ name: 'pinocchio', displayName: 'Pinocchio', description: 'Rigid body algorithms', loadState: 'error', error: 'Not installed' }),
];

describe('EngineSelector', () => {
  const defaultProps = {
    engines: mockEngines,
    selectedEngine: 'mujoco',
    onSelect: vi.fn(),
    onLoad: vi.fn(),
    onUnload: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders all engines', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByText('MuJoCo')).toBeInTheDocument();
      expect(screen.getByText('Drake')).toBeInTheDocument();
      expect(screen.getByText('Pinocchio')).toBeInTheDocument();
    });

    it('has radiogroup role with label', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByRole('radiogroup', { name: /physics engines/i })).toBeInTheDocument();
    });

    it('shows engine descriptions', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByText('High-performance physics for robotics')).toBeInTheDocument();
    });
  });

  describe('load states', () => {
    it('shows "Not loaded" for idle engines', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByText('Not loaded')).toBeInTheDocument();
    });

    it('shows "Loaded" for loaded engines', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByText(/Loaded/)).toBeInTheDocument();
    });

    it('shows error message for failed engines', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByText('Not installed')).toBeInTheDocument();
    });

    it('shows loading state with spinner', () => {
      const engines = [
        createEngine({ name: 'mujoco', displayName: 'MuJoCo', loadState: 'loading' }),
      ];

      render(<EngineSelector {...defaultProps} engines={engines} />);

      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });
  });

  describe('Load button', () => {
    it('shows Load button for idle engines', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByRole('button', { name: /load drake/i })).toBeInTheDocument();
    });

    it('shows Retry button for errored engines', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByRole('button', { name: /load pinocchio/i })).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });

    it('calls onLoad when Load button clicked', () => {
      const onLoad = vi.fn();
      render(<EngineSelector {...defaultProps} onLoad={onLoad} />);

      fireEvent.click(screen.getByRole('button', { name: /load drake/i }));
      expect(onLoad).toHaveBeenCalledWith('drake');
    });

    it('shows Unload button for loaded engines', () => {
      render(<EngineSelector {...defaultProps} />);

      expect(screen.getByRole('button', { name: /unload mujoco/i })).toBeInTheDocument();
    });

    it('calls onUnload when Unload button clicked', () => {
      const onUnload = vi.fn();
      const engines = [
        createEngine({ name: 'mujoco', displayName: 'MuJoCo', loadState: 'loaded' }),
      ];
      render(
        <EngineSelector
          {...defaultProps}
          engines={engines}
          selectedEngine={null}
          onUnload={onUnload}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /unload mujoco/i }));
      expect(onUnload).toHaveBeenCalledWith('mujoco');
    });

    it('disables Unload button on selected engine', () => {
      render(<EngineSelector {...defaultProps} selectedEngine="mujoco" />);

      const unloadBtn = screen.getByRole('button', { name: /unload mujoco/i });
      expect(unloadBtn).toBeDisabled();
    });
  });

  describe('selection', () => {
    it('only allows selecting loaded engines', () => {
      const onSelect = vi.fn();
      render(<EngineSelector {...defaultProps} onSelect={onSelect} />);

      // Drake is idle — clicking it should not call onSelect
      const drakeRadio = screen.getByRole('radio', { name: /drake.*not loaded/i });
      expect(drakeRadio).toBeDisabled();
    });

    it('calls onSelect for loaded engine click', () => {
      const onSelect = vi.fn();
      const engines = [
        createEngine({ name: 'mujoco', displayName: 'MuJoCo', loadState: 'loaded' }),
        createEngine({ name: 'drake', displayName: 'Drake', loadState: 'loaded' }),
      ];

      render(
        <EngineSelector
          {...defaultProps}
          engines={engines}
          selectedEngine="mujoco"
          onSelect={onSelect}
        />
      );

      fireEvent.click(screen.getByRole('radio', { name: /drake/i }));
      expect(onSelect).toHaveBeenCalledWith('drake');
    });

    it('shows aria-checked on selected engine', () => {
      render(<EngineSelector {...defaultProps} selectedEngine="mujoco" />);

      const selected = screen.getByRole('radio', { name: /mujoco/i });
      expect(selected).toHaveAttribute('aria-checked', 'true');
    });
  });

  describe('disabled state', () => {
    it('disables all interactions when disabled', () => {
      render(<EngineSelector {...defaultProps} disabled />);

      const radios = screen.getAllByRole('radio');
      radios.forEach((radio) => {
        expect(radio).toBeDisabled();
      });
    });
  });
});
