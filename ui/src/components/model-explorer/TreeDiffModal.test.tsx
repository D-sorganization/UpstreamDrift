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
});
