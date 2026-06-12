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

The Vite dev server listens on port `5180` by default. The browser UI expects the API server to be reachable on the same machine, and the WebSocket clients in `ui/src/api/client.ts` connect through `/api/ws/simulate/{engineType}`.

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

- `ui/src/api/` - client utilities and hooks that wrap REST and WebSocket flows.
- `ui/src/stores/` - Zustand state for engines, simulation state, and UI session data.
- `ui/src/integration/` - integration-focused UI tests.
- `ui/src-tauri/` - Rust packaging and native shell support for the desktop build.

## API contracts the UI depends on

The UI consumes HTTP routes defined under `src/api/routes/` and relies on WebSocket flows implemented in:

- `src/api/routes/simulation_ws.py`
- `src/api/routes/chat_ws.py`

Frontend call sites and hooks live under `ui/src/api/`. For transport details, start with `ui/src/api/client.ts`, which currently connects to `/api/ws/simulate/{engineType}`. The chat service protocol lives in `src/api/routes/chat_ws.py` under `/ws/chat/{session_id}`.

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
