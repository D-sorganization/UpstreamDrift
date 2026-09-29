/**
 * useModalA11y — keyboard accessibility for modal dialogs (issue #7438).
 *
 * Attach the returned ref to the dialog container. While `isOpen`:
 * - the previously focused element is saved and restored on close;
 * - focus moves into the dialog (first focusable element, else the container);
 * - Escape closes the dialog (listener scoped to the dialog so stacked dialogs
 *   don't leak Escape to each other);
 * - Tab / Shift+Tab wrap at the edges (focus trap), keeping focus inside.
 *
 * WCAG 2.1.1 (Keyboard) and 2.4.3 (Focus Order).
 */

import { useCallback, useEffect, useRef } from 'react';

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function useModalA11y<T extends HTMLElement = HTMLDivElement>(
  isOpen: boolean,
  onClose: () => void,
) {
  const containerRef = useRef<T | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  const getFocusable = useCallback((): HTMLElement[] => {
    const root = containerRef.current;
    if (!root) {
      return [];
    }
    // Exclude elements hidden via the `hidden` attribute or an aria-hidden
    // ancestor; do NOT rely on offsetParent/layout, which jsdom never computes
    // (that would hide every element under test).
    return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
      (el) =>
        !el.hasAttribute('hidden') &&
        el.closest('[aria-hidden="true"]') === null,
    );
  }, []);

  // Save + restore focus and move focus into the dialog on open.
  useEffect(() => {
    if (!isOpen) {
      return;
    }
    previouslyFocusedRef.current =
      document.activeElement as HTMLElement | null;

    const focusable = getFocusable();
    if (focusable.length > 0) {
      focusable[0].focus();
    } else if (containerRef.current) {
      containerRef.current.tabIndex = -1;
      containerRef.current.focus();
    }

    return () => {
      previouslyFocusedRef.current?.focus?.();
    };
  }, [isOpen, getFocusable]);

  // Escape to close + focus trap, scoped to the dialog container.
  useEffect(() => {
    const node = containerRef.current;
    if (!isOpen || !node) {
      return;
    }

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key !== 'Tab') {
        return;
      }
      const focusable = getFocusable();
      if (focusable.length === 0) {
        e.preventDefault();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !node.contains(active))) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    node.addEventListener('keydown', handleKeyDown);
    return () => node.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, getFocusable]);

  return containerRef;
}
