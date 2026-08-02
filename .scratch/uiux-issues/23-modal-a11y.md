## Problem

Modal dialogs are not keyboard-accessible (WCAG 2.1.1, 2.4.3):

1. **TreeDiffModal** — `ui/src/components/model-explorer/TreeDiffModal.tsx` (~line 29): the overlay div has no `role="dialog"`, no `aria-modal="true"`, no `aria-labelledby`; Escape does not close it; focus is not moved into the dialog on open nor restored on close; Tab walks the background page.
2. **HelpPanel** — `ui/src/components/ui/HelpPanel.tsx` (~line 151): has `role="dialog"`, `aria-modal`, and Escape handling, but no focus trap — Tab escapes to the background — and initial focus relies on a 100ms timeout.

## Fix

Implement once, use in both (and any future modal):

1. Create `ui/src/utils/useModalA11y.ts` (or a `<ModalShell>` component) providing:
   - on open: save `document.activeElement`, focus the first focusable element (or the dialog container with `tabIndex={-1}`);
   - Escape key closes (listener on the dialog, not window, to avoid leaking between stacked dialogs);
   - focus trap: on Tab/Shift+Tab at the edges, wrap to the other end (query `a, button, input, select, textarea, [tabindex]:not([tabindex="-1"])` within the dialog);
   - on close: restore focus to the saved element.
2. TreeDiffModal markup:
   ```tsx
   <div className="fixed inset-0 ... z-50" role="dialog" aria-modal="true" aria-labelledby="tree-diff-title">
     <h2 id="tree-diff-title">Model Comparison</h2>
   ```
   Also mark the decorative 📊 emoji (~line 34) `aria-hidden="true"`.
3. Apply the same hook to HelpPanel, replacing the timeout-based focus.
4. Tests: Escape closes both; Tab from last element wraps to first; focus returns to the opener button on close.

## Acceptance criteria

- Both dialogs fully operable with keyboard only; tests pass; axe (or vitest-axe) reports no dialog violations.

Part of the UI/UX overhaul epic (see tracking issue).
