/**
 * Responsive WorkspaceShell tests (UI/UX #7415).
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { WorkspaceShell } from './WorkspaceShell';

describe('WorkspaceShell', () => {
  it('renders left, right, main, and bottom content', () => {
    render(
      <WorkspaceShell
        leftPanel={<div>LEFT</div>}
        rightPanel={<div>RIGHT</div>}
        bottomPanel={<div>BOTTOM</div>}
      >
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    // Docked panels render their content (duplicated into the hidden drawer too).
    expect(screen.getAllByText('LEFT').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('RIGHT').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('MAIN')).toBeInTheDocument();
    expect(screen.getByText('BOTTOM')).toBeInTheDocument();
  });

  it('keeps the main element shrinkable for canvas children (min-w-0/min-h-0)', () => {
    render(
      <WorkspaceShell leftPanel={<div>L</div>}>
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    const main = screen.getByRole('main');
    expect(main.className).toContain('min-w-0');
    expect(main.className).toContain('min-h-0');
  });

  it('exposes the skip-link target on main (#7441)', () => {
    render(
      <WorkspaceShell leftPanel={<div>L</div>}>
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    const main = screen.getByRole('main');
    expect(main).toHaveAttribute('id', 'main-content');
    expect(main).toHaveAttribute('tabindex', '-1');
  });

  it('opens the left drawer when its toggle is clicked', () => {
    render(
      <WorkspaceShell leftPanel={<div>LEFTPANEL</div>} leftPanelLabel="Controls">
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    // No dialog before opening.
    expect(screen.queryByRole('dialog')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Controls' }));
    const dialog = screen.getByRole('dialog', { name: 'Controls' });
    expect(dialog).toBeInTheDocument();
  });

  it('gives drawer toggles a 44px minimum touch target (#14279)', () => {
    render(
      <WorkspaceShell leftPanel={<div>L</div>} leftPanelLabel="Controls">
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    const toggle = screen.getByRole('button', { name: 'Controls' });
    expect(toggle.className).toContain('min-h-[44px]');
    expect(toggle.className).toContain('min-w-[44px]');
  });

  it('lets a page override the docked left panel width', () => {
    render(
      <WorkspaceShell leftPanel={<div>L</div>} leftPanelWidthClass="lg:w-96">
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    const aside = document.querySelector('aside');
    expect(aside?.className).toContain('lg:w-96');
    expect(aside?.className).not.toContain('lg:w-72');
  });

  it('closes the drawer via the close button', () => {
    render(
      <WorkspaceShell rightPanel={<div>RP</div>} rightPanelLabel="Details">
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Details' }));
    expect(screen.getByRole('dialog', { name: 'Details' })).toBeInTheDocument();
    // There are two "Close Details" affordances (backdrop + button); clicking
    // either dismisses the drawer.
    fireEvent.click(screen.getAllByRole('button', { name: 'Close Details' })[0]);
    expect(screen.queryByRole('dialog')).toBeNull();
  });

  it('omits a side entirely when its panel prop is not provided', () => {
    render(
      <WorkspaceShell leftPanel={<div>ONLYLEFT</div>}>
        <div>MAIN</div>
      </WorkspaceShell>,
    );
    // Only the left toggle exists; no right toggle.
    expect(
      screen.getByRole('button', { name: 'Controls' }),
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Details' })).toBeNull();
  });
});
