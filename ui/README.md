# UpstreamDrift UI

The UpstreamDrift UI is a React 19 + TypeScript + Vite application with an optional Tauri desktop shell. It talks to the local-first Python API over REST and WebSockets, renders 3D scenes with `@react-three/fiber`, and keeps shared client state in Zustand stores.

## Local development

Start the Python API from the repository root:

```bash
python start_api_server.py --port 8000
```

In a second terminal, start the UI from `ui/`:

```bash
cd ui
npm install
npm run dev
```

The Vite dev server listens on port `5180` by default and proxies `/api` and
`/api/ws` to the API on port `8000` — the same port documented above,
`launch_upstream_drift.py --port` defaults to, and `BACKEND_PORT` in
`ui/src/api/backend.ts` declares. Previously the proxy pointed at `8001`, so
following these instructions left the dashboard on
`HTTP 500 — /api/launcher/manifest` (issue #8076).

If you deliberately run the API on another port, start Vite with a matching
override rather than editing the config:

```bash
VITE_API_PORT=8001 npm run dev
```

The proxy table lives in `ui/src/config/devProxy.ts` and is covered by
`ui/src/test/vite-proxy-contract.test.ts`.

The browser UI expects the API server to be reachable on the same machine, and the WebSocket clients in `ui/src/api/client.ts` connect through `/api/ws/simulate/{engineType}`.

## Architecture at a glance

- `ui/src/pages/` - route-level screens such as the dashboard and simulation views.
- `ui/src/components/` - reusable UI building blocks and domain components, including `visualization/Scene3D.tsx`.
- `ui/src/components/ui/` - shared design primitives (`Button`, `Input`, `Select`, `Badge`, `Card`). See "Design system" below.

## Design system

All new buttons, inputs, and selects **must** use the shared primitives in
`components/ui/` (`Button`, `Input`, `Select`, `Badge`, `Card`) instead of
hand-rolled Tailwind strings, so paddings, radii, focus rings, and disabled
states stay consistent (UI/UX #7420). For one-off focusable elements that can't
use a primitive, apply the `.focus-ring` component class from `index.css`.

Color rules (UI/UX #7421), enforced by `utils/colorGuard.test.ts`:

- **Neutrals: `gray-*` only** (no `slate-`, `zinc-`, `neutral-`).
- **Primary accent: `blue-*`** (`blue-600` / `blue-700` hover). Green is for
  run/start/success, red for destructive actions.
- **Status semantics:** success `green-*`, error `red-*`, warning `amber-*`
  (not `yellow-*`), info `blue-*`. Semantic aliases (`primary`, `success`,
  `warning`, `danger`) are defined in `tailwind.config.js`.

Intentional exceptions to the color guard live in
`utils/colorGuard.allowlist.json` with a documented reason.

### Typography scale (UI/UX #7422)

Use the shared heading/label classes from `index.css` instead of ad-hoc
`text-* font-*` combinations so identical-rank headings match across pages:

| Class              | Use for                                  | Element |
| ------------------ | ---------------------------------------- | ------- |
| `.heading-page`    | the single top-level page title          | `h1`    |
| `.heading-section` | a section heading                        | `h2`    |
| `.heading-sub`     | a sub-section heading                    | `h3`    |
| `.label-overline`  | uppercase overline labels above controls | `label` |

Each page should have exactly one `h1` and must not skip heading ranks (an
`sr-only` `h1` is fine where the visible hierarchy starts at `h2`).

### Theming tokens (UI/UX #7423)

Theming is class-based: `tailwind.config.js` sets `darkMode: 'class'` and the
root `<html>` carries `class="dark"`, so the default appearance is the dark
theme. The runtime-themeable `--sidekick-color-*` CSS variables in `index.css`
are the source of truth; a `:root:not(.dark)` block provides a light theme,
proving the plumbing works when the root class is flipped.

Token-backed Tailwind utilities map onto those variables — prefer them in new
code so a re-theme is a variable swap, not a class-string edit:

| Utility               | Variable                          |
| --------------------- | --------------------------------- |
| `bg-canvas`           | `--sidekick-color-canvas`         |
| `bg-surface`          | `--sidekick-color-surface`        |
| `bg-surface-raised`   | `--sidekick-color-surface-raised` |
| `border-token-border` | `--sidekick-color-border`         |
| `text-token-text`     | `--sidekick-color-text`           |
| `text-token-muted`    | `--sidekick-color-text-muted`     |
| `text-token-subtle`   | `--sidekick-color-text-subtle`    |

- `ui/src/api/` - client utilities and hooks that wrap REST and WebSocket flows.
- `ui/src/stores/` - Zustand state for engines, simulation state, and UI session data.
- `ui/src/integration/` - integration-focused UI tests.
- `ui/src-tauri/` - Rust packaging and native shell support for the desktop build.

## API contracts the UI depends on

The UI consumes HTTP routes defined under `src/api/routes/` and relies on WebSocket flows implemented in:

- `src/api/routes/simulation_ws.py`
- `src/api/routes/chat_ws.py`

Frontend call sites and hooks live under `ui/src/api/`. For transport details, start with `ui/src/api/client.ts`, which currently connects to `/api/ws/simulate/{engineType}`. The chat service protocol lives in `src/api/routes/chat_ws.py` under `/ws/chat/{session_id}`.

### Generated API types (issue #7447)

`ui/src/api/generated/types.ts` is **generated** from the backend OpenAPI
contract (the Pydantic models in `src/api/models/` plus route response
models) and checked in. **Never hand-write a TypeScript interface for a
payload that exists in `src/api/models/`** — import it from
`./generated/types` instead (see `EngineStatus` in `ui/src/api/client.ts` and
the launcher-manifest types in `ui/src/api/useLauncherManifest.ts` for the
pattern). If an endpoint you consume has no Pydantic response model yet, add
one in `src/api/models/responses.py` and attach it to the route, then
regenerate.

Regenerate after any contract change:

```bash
python scripts/generate_ui_api_types.py
```

A pytest freshness gate (`tests/api/test_generated_ui_api_types.py`)
regenerates the file in CI and fails if the committed copy is stale, so the
generated contract can never drift silently. Do not edit the generated file
by hand.

### Runtime response validation (issue #7165)

The backend and frontend ship separately (Docker vs Tauri vs dev), so response
shape skew is a _when_, not an _if_. `apiFetch<T>` is a compile-time-only
assertion, so a malformed payload would otherwise be stored as-is and crash deep
inside a component render. Boundary hooks therefore validate at runtime via
`apiFetchParsed(path, parseFn)` (`ui/src/api/fetch.ts`) with parser functions in
`ui/src/api/schemas.ts`; an invalid payload becomes the hook's `error` state
instead of a render-time `TypeError`.

**Decision:** validation uses small hand-rolled type guards rather than a
runtime-schema dependency (e.g. `zod`). The validated types are the existing
TypeScript interfaces and each `parseX` returns that same interface, so the
compile-time and runtime contracts share one source of truth without an added
dependency. New validated boundaries should follow this pattern: add a
`parseX` to `schemas.ts` and fetch through `apiFetchParsed`.

### WebSocket URL resolution (issue #7166)

There is one URL-resolution path for HTTP and WebSockets: components ask
`getApiBase()` in `ui/src/api/backend.ts` and never compute origins themselves.
`VITE_API_URL` is the single override. Chat (`resolveChatUrl` in
`ChatPanel.tsx`) and the simulation socket both derive their `ws(s)://` URL from
that base.

## Desktop build

Install the Rust toolchain and the Tauri build prerequisites for your platform, then run:

```bash
cd ui
npm install
npm run tauri:build
```

For local desktop iteration, use `npm run tauri:dev`.

## Testing and linting

From `ui/`:

```bash
npm run test
npm run lint
```

`npm run test` uses Vitest. Integration-oriented coverage lives under `ui/src/integration/`.

## Contributing

- Repository contribution workflow: [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
- Agent and repository policy: [`../CLAUDE.md`](../CLAUDE.md)
- UI parity assessment notes: [`../docs/assessments/issues/REACT_UI_PARITY_ISSUES.md`](../docs/assessments/issues/REACT_UI_PARITY_ISSUES.md)

The real ESLint configuration for this package lives in `ui/eslint.config.js`; do not copy guidance from the stock Vite scaffold.

## Known gaps

This README is a narrow developer entry point, not the full UI handbook. Active UI follow-up work is being tracked under `#3160`, including the chat UI, scaffolding pages, HelpPanel content, and model-authoring gaps called out in that track.
