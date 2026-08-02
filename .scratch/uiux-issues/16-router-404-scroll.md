## Problem (verified in code)

`ui/src/App.tsx` lines 27–50: the `<Routes>` block has **no catch-all route**. Any mistyped or stale deep link (e.g. `/tools/model` or an old bookmarked path) renders a completely blank page — no message, no way back. Additionally there is no scroll restoration: navigating between pages preserves the previous scroll offset on pages that scroll.

## Fix

1. Create `ui/src/pages/NotFound.tsx`: dark-themed card matching the app (`bg-gray-900` page, `bg-gray-800` card) showing "Page not found", the attempted path (`useLocation().pathname`), and a primary-button link to `/` plus a link to `/simulation`.
2. Register it as the last route in `App.tsx`:
   ```tsx
   <Route path="*" element={<NotFoundPage />} />
   ```
3. Add `ui/src/utils/ScrollToTop.tsx`:
   ```tsx
   export function ScrollToTop() {
     const { pathname } = useLocation();
     useEffect(() => {
       window.scrollTo(0, 0);
     }, [pathname]);
     return null;
   }
   ```
   Mount it directly inside `<BrowserRouter>`. (Most pages are `h-screen overflow-hidden` shells, but Dashboard/launcher and future scrollable pages need this.)
4. Tests: rendering router at an unknown path shows the NotFound content; known routes unaffected.

## Acceptance criteria

- Navigating to any undefined path shows the branded 404 page with working links.
- Navigation always lands at the top of the new page.

Part of the UI/UX overhaul epic (see tracking issue).
