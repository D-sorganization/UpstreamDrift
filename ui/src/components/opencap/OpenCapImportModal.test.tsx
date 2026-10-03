import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { OpenCapImportModal } from './OpenCapImportModal';

describe('OpenCapImportModal Component', () => {
  const mockMetadata = {
    session_dir: '/path/to/opencap_session',
    trials: ['neutral', 'swing1', 'swing2'],
    subject: {
      mass_kg: 79.5,
      height_m: 1.82,
      sex: 'm',
      opensim_model: 'LaiUhlrich2022',
      subject_id: 'sub-01',
    },
    model_file: '/path/to/opencap_session/OpenSimData/Model/LaiUhlrich2022_scaled.osim',
    kinematics_trials: ['swing1', 'swing2'],
    notes: [],
  };

  it('does not render when isOpen is false', () => {
    render(
      <OpenCapImportModal
        isOpen={false}
        onClose={vi.fn()}
        metadata={mockMetadata}
        onImport={vi.fn()}
      />
    );
    expect(screen.queryByText(/import opencap session/i)).not.toBeInTheDocument();
  });

  it('renders modal with trials list when open', () => {
    render(
      <OpenCapImportModal
        isOpen={true}
        onClose={vi.fn()}
        metadata={mockMetadata}
        onImport={vi.fn()}
      />
    );

    expect(screen.getByText(/import opencap session/i)).toBeInTheDocument();
    expect(screen.getByText('neutral')).toBeInTheDocument();
    expect(screen.getByText('swing1')).toBeInTheDocument();
    expect(screen.getByText('swing2')).toBeInTheDocument();
  });

  it('allows user to select a trial and dispatches onImport', () => {
    const handleImport = vi.fn();
    render(
      <OpenCapImportModal
        isOpen={true}
        onClose={vi.fn()}
        sessionDir="/path/to/opencap_session"
        metadata={mockMetadata}
        onImport={handleImport}
      />
    );

    // Select 'swing2'
    const trialButton = screen.getByRole('button', { name: /swing2/i });
    fireEvent.click(trialButton);

    // Click Import button
    const importButton = screen.getByRole('button', { name: /^import session$/i });
    fireEvent.click(importButton);

    expect(handleImport).toHaveBeenCalledTimes(1);
    expect(handleImport).toHaveBeenCalledWith({
      sessionDir: '/path/to/opencap_session',
      selectedTrial: 'swing2',
      metadata: mockMetadata,
    });
  });

  it('calls onClose when Cancel or Close is clicked', () => {
    const handleClose = vi.fn();
    render(
      <OpenCapImportModal
        isOpen={true}
        onClose={handleClose}
        metadata={mockMetadata}
        onImport={vi.fn()}
      />
    );

    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    fireEvent.click(cancelButton);
    expect(handleClose).toHaveBeenCalled();
  });

  it('displays error state when provided', () => {
    render(
      <OpenCapImportModal
        isOpen={true}
        onClose={vi.fn()}
        error="Directory has no valid marker trials"
        onImport={vi.fn()}
      />
    );

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText(/directory has no valid marker trials/i)).toBeInTheDocument();
  });
});
