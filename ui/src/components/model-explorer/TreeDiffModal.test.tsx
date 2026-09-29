import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TreeDiffModal } from './TreeDiffModal';
import { TreeDiff } from '@/utils/frankensteinTree';

describe('TreeDiffModal Component', () => {
  const mockDiff: TreeDiff = {
    added: ['new_link'],
    removed: ['old_joint'],
    modified: [
      {
        id: 'common_node',
        changes: [
          {
            field: 'mass',
            sourceVal: 10,
            targetVal: 15,
          },
        ],
      },
    ],
  };

  it('does not render when isOpen is false', () => {
    render(
      <TreeDiffModal
        isOpen={false}
        onClose={() => {}}
        sourceModelName="Source"
        targetModelName="Target"
        diff={mockDiff}
      />
    );
    expect(screen.queryByText(/model comparison/i)).not.toBeInTheDocument();
  });

  it('renders all diff details when isOpen is true', () => {
    const handleClose = vi.fn();
    render(
      <TreeDiffModal
        isOpen={true}
        onClose={handleClose}
        sourceModelName="Source Robot"
        targetModelName="Target Robot"
        diff={mockDiff}
      />
    );

    expect(screen.getByText(/Model Comparison/i)).toBeInTheDocument();
    expect(screen.getByText(/new_link/i)).toBeInTheDocument();
    expect(screen.getByText(/old_joint/i)).toBeInTheDocument();
    expect(screen.getByText(/common_node/i)).toBeInTheDocument();
    expect(screen.getByText(/mass/i)).toBeInTheDocument();
    expect(screen.getByText(/10/i)).toBeInTheDocument();

    const closeBtn = screen.getByLabelText(/close modal/i);
    fireEvent.click(closeBtn);
    expect(handleClose).toHaveBeenCalled();
  });

  it('exposes dialog semantics and a labelled title (#7438)', () => {
    render(
      <TreeDiffModal
        isOpen
        onClose={vi.fn()}
        sourceModelName="S"
        targetModelName="T"
        diff={mockDiff}
      />,
    );
    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    const title = document.getElementById(
      dialog.getAttribute('aria-labelledby')!,
    );
    expect(title).toHaveTextContent('Model Comparison');
  });

  it('closes on Escape (#7438)', () => {
    const onClose = vi.fn();
    render(
      <TreeDiffModal
        isOpen
        onClose={onClose}
        sourceModelName="S"
        targetModelName="T"
        diff={mockDiff}
      />,
    );
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('traps Tab focus within the dialog (#7438)', () => {
    render(
      <TreeDiffModal
        isOpen
        onClose={vi.fn()}
        sourceModelName="S"
        targetModelName="T"
        diff={mockDiff}
      />,
    );
    const dialog = screen.getByRole('dialog');
    const focusable = dialog.querySelectorAll('button');
    const last = focusable[focusable.length - 1] as HTMLElement;
    last.focus();
    fireEvent.keyDown(dialog, { key: 'Tab' });
    // Wrapped back to the first focusable (the close button).
    expect(focusable[0]).toHaveFocus();
  });
});
