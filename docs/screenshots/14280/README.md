# Dashboard Header on Phones (#14280)

Before (main) and after (this branch) screenshots of the Dashboard (`/`). They
were captured with headless Chromium against the Vite dev server, with
`/api/launcher/manifest` stubbed from `src/config/launcher_manifest.json`.

## Measurements

"Clipped" counts the header actions and Task Workspaces links (10 in total)
whose bounding box extends past the viewport edge. These links can't be
reached because the page layout is `overflow-hidden`.

| Viewport | Before Clipped                                | After Clipped | Header / Nav Height After |
| -------- | --------------------------------------------- | ------------- | ------------------------- |
| 390×844  | 5 (Help and four workspace links)             | 0             | 117 / 115 px              |
| 768×1024 | 1 (Developer & Research, right edge at x=877) | 0             | 79 / 81 px                |
| 1280×800 | 0                                             | 0             | 79 / 47 px (unchanged)    |

## Files

- `before-dashboard-{phone,tablet,desktop}.png`
- `after-dashboard-{phone,tablet,desktop}.png`
