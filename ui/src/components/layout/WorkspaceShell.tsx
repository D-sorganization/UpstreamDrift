import { useState } from 'react';
import type { ReactNode } from 'react';

export interface WorkspaceShellProps {
  /** Left sidebar content (controls). Omit for a two-column layout. */
  leftPanel?: ReactNode;
  /** Right sidebar content (live data / inspectors). Omit to hide. */
  rightPanel?: ReactNode;
  /** Optional full-width strip below the main area (e.g. a plot strip). */
  bottomPanel?: ReactNode;
  /** Main content (3D viewport, canvas, etc.). */
  children: ReactNode;
  /** Accessible labels for the drawer toggle buttons. */
  leftPanelLabel?: string;
  rightPanelLabel?: string;
  /** Extra classes for the outer container (rarely needed). */
  className?: string;
}

const ASIDE_BASE =
  'flex flex-col flex-shrink-0 bg-gray-800 overflow-y-auto min-h-0';

/**
 * Canonical responsive workspace frame (UI/UX #7415).
 *
 * Replaces the bespoke `w-80`/`w-72` fixed-width aside markup that every
 * multi-panel page hand-rolled. Above the `lg` breakpoint it renders the
 * classic three-column layout; below it the side panels collapse into
 * toggleable overlay drawers so every control stays reachable on narrow
 * windows and Tauri popouts.
 *
 * The `min-w-0`/`min-h-0` on the `<main>` element is load-bearing: without it
 * a flex child containing a WebGL canvas refuses to shrink below its content
 * size and overflows (see #7416).
 */
export function WorkspaceShell({
  leftPanel,
  rightPanel,
  bottomPanel,
  children,
  leftPanelLabel = 'Controls',
  rightPanelLabel = 'Details',
  className = '',
}: WorkspaceShellProps) {
  const [leftOpen, setLeftOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(false);

  const hasLeft = leftPanel != null;
  const hasRight = rightPanel != null;

  return (
    <div
      className={`flex flex-col h-screen bg-gray-900 overflow-hidden ${className}`.trim()}
    >
      {/* Mobile/tablet top bar with drawer toggles (hidden at lg+). */}
      {(hasLeft || hasRight) && (
        <div className="flex lg:hidden items-center justify-between gap-2 bg-gray-800 border-b border-gray-700 px-3 py-2 flex-shrink-0">
          {hasLeft ? (
            <DrawerToggle
              label={leftPanelLabel}
              expanded={leftOpen}
              onClick={() => setLeftOpen(true)}
            />
          ) : (
            <span />
          )}
          {hasRight ? (
            <DrawerToggle
              label={rightPanelLabel}
              expanded={rightOpen}
              onClick={() => setRightOpen(true)}
            />
          ) : (
            <span />
          )}
        </div>
      )}

      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left: docked column at lg+; overlay drawer below. */}
        {hasLeft && (
          <>
            <aside
              className={`${ASIDE_BASE} border-r border-gray-700 hidden lg:flex lg:w-72 xl:w-80`}
            >
              {leftPanel}
            </aside>
            <DrawerOverlay
              open={leftOpen}
              side="left"
              label={leftPanelLabel}
              onClose={() => setLeftOpen(false)}
            >
              {leftPanel}
            </DrawerOverlay>
          </>
        )}

        {/* Main: must be allowed to shrink (#7416). The id is the skip-link
            target (#7441) so keyboard users can jump past the sidebars. */}
        <main
          id="main-content"
          tabIndex={-1}
          className="flex-1 flex flex-col min-w-0 min-h-0 relative"
        >
          <div className="flex-1 flex flex-col min-w-0 min-h-0">{children}</div>
          {bottomPanel != null && (
            <div className="flex-shrink-0">{bottomPanel}</div>
          )}
        </main>

        {/* Right: docked column at xl+; overlay drawer below. */}
        {hasRight && (
          <>
            <aside
              className={`${ASIDE_BASE} border-l border-gray-700 hidden xl:flex xl:w-72`}
            >
              {rightPanel}
            </aside>
            <DrawerOverlay
              open={rightOpen}
              side="right"
              label={rightPanelLabel}
              onClose={() => setRightOpen(false)}
            >
              {rightPanel}
            </DrawerOverlay>
          </>
        )}
      </div>
    </div>
  );
}

interface DrawerToggleProps {
  label: string;
  expanded: boolean;
  onClick: () => void;
}

function DrawerToggle({ label, expanded, onClick }: DrawerToggleProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-expanded={expanded}
      className="inline-flex items-center gap-1.5 px-2 py-1 text-xs font-medium rounded-md bg-gray-700 hover:bg-gray-600 text-gray-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
    >
      <span aria-hidden="true">☰</span>
      {label}
    </button>
  );
}

interface DrawerOverlayProps {
  open: boolean;
  side: 'left' | 'right';
  label: string;
  onClose: () => void;
  children: ReactNode;
}

function DrawerOverlay({
  open,
  side,
  label,
  onClose,
  children,
}: DrawerOverlayProps) {
  if (!open) return null;
  const sideClasses =
    side === 'left'
      ? 'left-0 border-r border-gray-700'
      : 'right-0 border-l border-gray-700';
  return (
    <div className="lg:hidden fixed inset-0 z-50 flex" role="dialog" aria-modal="true" aria-label={label}>
      {/* Backdrop */}
      <button
        type="button"
        aria-label={`Close ${label}`}
        onClick={onClose}
        className="absolute inset-0 bg-black/60"
      />
      <aside
        className={`${ASIDE_BASE} absolute inset-y-0 w-72 max-w-[85vw] ${sideClasses} shadow-xl`}
      >
        <div className="flex items-center justify-between px-3 py-2 border-b border-gray-700 flex-shrink-0">
          <span className="text-sm font-semibold text-gray-200">{label}</span>
          <button
            type="button"
            onClick={onClose}
            aria-label={`Close ${label}`}
            className="px-2 py-0.5 text-gray-400 hover:text-gray-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 rounded"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </div>
        {children}
      </aside>
    </div>
  );
}
