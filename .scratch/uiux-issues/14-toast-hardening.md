## Problem

`ui/src/components/ui/Toast.tsx` has no stacking limits or deduplication (~lines 82–85): every `showToast` appends to the list. Repeated failures (e.g. backend down + retries) stack unbounded toasts that can fill the screen. Additionally all toasts render with the same politeness, so screen-reader users get errors announced lazily (or not at all if dismissed early).

## Fix

1. Cap the visible stack: `MAX_TOASTS = 5`; drop the oldest when exceeded:
   ```tsx
   setToasts((prev) => [...prev, t].slice(-MAX_TOASTS));
   ```
2. Deduplicate: if a toast with identical `message` + `type` is already visible, reset its timer (and optionally show a `×N` count badge) instead of appending a duplicate.
3. Accessibility: render each toast with `role="alert"`/`aria-live="assertive"` for `type === 'error'` and `role="status"`/`aria-live="polite"` otherwise, plus `aria-atomic="true"`.
4. Ensure dismiss timers are cleared on unmount (verify the existing `useEffect` cleanup covers manual dismissal paths).
5. Tests: 10 rapid errors → ≤5 toasts; duplicate message extends rather than appends; error toast has `role="alert"`.

## Acceptance criteria

- Stack never exceeds 5; duplicates coalesce; tests pass.

Part of the UI/UX overhaul epic (see tracking issue).
