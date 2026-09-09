/**
 * Unsaved-changes guard for pages that hold edits in local state (#8892).
 *
 * Grepping `beforeunload|useBlocker|unsaved` across `ui/src` returned zero
 * non-test hits before this file existed: no page tracked dirty state, none
 * blocked navigation, none guarded unload. Changing settings and clicking the
 * back arrow discarded everything with no signal.
 *
 * **Why not React Router's `useBlocker`.** `useBlocker` requires a data router
 * (`createBrowserRouter`). `App.tsx` mounts a plain `<BrowserRouter>` with
 * `<Routes>`, where `useBlocker` throws. Migrating the app to a data router is
 * a separate change with its own blast radius, so this hook guards the two
 * exits a page actually owns:
 *
 * 1. `guardedNavigate` — wrap every in-app link/button on the page so it
 *    confirms before leaving.
 * 2. `beforeunload` — the browser's own prompt for tab close, reload, and
 *    external navigation, registered only while dirty so it never nags on a
 *    pristine page.
 *
 * When the app moves to a data router, `guardedNavigate` is the seam to swap
 * for `useBlocker`; the dirty-state half of the contract stays as it is.
 */

import { useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router';

export interface UnsavedChangesGuard {
  /** Navigate to `to`, confirming first when there are unsaved edits. */
  guardedNavigate: (to: string) => void;
  /** Run `action` only if the user accepts losing unsaved edits. */
  confirmDiscard: () => boolean;
}

export const DEFAULT_DISCARD_MESSAGE =
  'You have unsaved changes. Leave this page and discard them?';

/**
 * @param isDirty  Whether the page currently holds unsaved edits.
 * @param message  Confirmation text for in-app navigation.
 */
export function useUnsavedChangesGuard(
  isDirty: boolean,
  message: string = DEFAULT_DISCARD_MESSAGE,
): UnsavedChangesGuard {
  const navigate = useNavigate();

  // Browser-level guard: tab close, reload, external navigation. Registered
  // only while dirty, so a pristine page never triggers the native prompt.
  useEffect(() => {
    if (!isDirty) return;
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      // Legacy browsers need returnValue set; the string itself is ignored
      // by every current browser, which shows its own wording.
      event.returnValue = '';
      return '';
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [isDirty]);

  const confirmDiscard = useCallback((): boolean => {
    if (!isDirty) return true;
    return window.confirm(message);
  }, [isDirty, message]);

  const guardedNavigate = useCallback(
    (to: string) => {
      if (!confirmDiscard()) return;
      void navigate(to);
    },
    [confirmDiscard, navigate],
  );

  return { guardedNavigate, confirmDiscard };
}
