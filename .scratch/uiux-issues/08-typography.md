## Problem

Headings and labels have no enforced scale, so identical-rank headings differ across pages:

- `ui/src/pages/Simulation.tsx` ~241: `<h2 className="text-xl font-bold">Golf Suite</h2>`
- `ui/src/pages/DataExplorer.tsx` ~385: `<h2 className="text-lg font-bold">Data Explorer</h2>`
- `ui/src/components/ui/HelpPanel.tsx` ~267/354: `text-xl font-bold`
- Section labels vary among `text-xs font-semibold uppercase tracking-wider` (ParameterPanel), `text-sm font-medium`, `text-lg font-semibold`.

Heading levels also skip ranks on some pages (sidebar `h2` followed by `h3` with no page `h1`), which hurts both visual hierarchy and screen-reader navigation.

## Fix

1. Define component classes in `ui/src/index.css`:
   ```css
   @layer components {
     .heading-page {
       @apply text-xl font-bold text-white;
     } /* one per page, h1 */
     .heading-section {
       @apply text-base font-semibold text-gray-100;
     } /* h2 */
     .heading-sub {
       @apply text-sm font-semibold text-gray-200;
     } /* h3 */
     .label-overline {
       @apply text-xs font-semibold text-gray-400 uppercase tracking-wider;
     }
   }
   ```
2. Sweep all pages/components: every page gets exactly one `h1.heading-page` (visible or `sr-only`), sections use `h2.heading-section`, sub-sections `h3.heading-sub`, panel overline labels use `.label-overline`. No skipped heading ranks.
3. Replace the ad-hoc combinations listed above; find the rest with `grep -rn "<h[1-4]" ui/src`.

## Acceptance criteria

- Every page has exactly one h1; heading ranks never skip.
- All headings/labels use the shared classes; `grep -rn "text-xl font-bold\|text-lg font-bold" ui/src` returns no raw instances outside `index.css`.

Part of the UI/UX overhaul epic (see tracking issue).
