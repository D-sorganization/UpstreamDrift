/**
 * useModalA11y tests (issue #7438).
 */

import { useState } from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useModalA11y } from './useModalA11y';

function Dialog({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  const ref = useModalA11y<HTMLDivElement>(isOpen, onClose);
  if (!isOpen) return null;
  return (
    <div ref={ref} role="dialog" aria-label="test">
      <button>first</button>
      <button>middle</button>
      <button>last</button>
    </div>
  );
}

describe('useModalA11y', () => {
  it('moves focus to the first focusable element on open', () => {
    render(<Dialog isOpen onClose={vi.fn()} />);
    expect(screen.getByText('first')).toHaveFocus();
  });

  it('closes on Escape', () => {
    const onClose = vi.fn();
    render(<Dialog isOpen onClose={onClose} />);
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('wraps focus from the last element to the first on Tab', () => {
    render(<Dialog isOpen onClose={vi.fn()} />);
    const last = screen.getByText('last');
    last.focus();
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Tab' });
    expect(screen.getByText('first')).toHaveFocus();
  });

  it('wraps focus from the first element to the last on Shift+Tab', () => {
    render(<Dialog isOpen onClose={vi.fn()} />);
    const first = screen.getByText('first');
    first.focus();
    fireEvent.keyDown(screen.getByRole('dialog'), {
      key: 'Tab',
      shiftKey: true,
    });
    expect(screen.getByText('last')).toHaveFocus();
  });

  it('restores focus to the opener when closed', () => {
    function Harness() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button onClick={() => setOpen(true)}>opener</button>
          <Dialog isOpen={open} onClose={() => setOpen(false)} />
        </>
      );
    }
    render(<Harness />);
    const opener = screen.getByText('opener');
    opener.focus();
    fireEvent.click(opener);
    // dialog open, first focused
    expect(screen.getByText('first')).toHaveFocus();
    fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' });
    expect(opener).toHaveFocus();
  });
});
