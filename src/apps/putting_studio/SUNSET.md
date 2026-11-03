# Sunset Criterion: `js/putting_physics.js`

`js/putting_physics.js` is a hand-kept copy of putting physics.
The single source of truth is the Python core at
`src/shared/python/putting_dynamics/`; the PyQt6 tool and this HTML
studio are both clients of it, not peers. The copy exists because
the studio opens from `file://` with no build step and no runtime CDN
dependencies, so it cannot `import` the Python package directly.

## Sunset Criterion and Evaluation (#14200)

This copy was evaluated against generated and bridged alternatives
under the measured offline budget in `tests/apps/putting_studio/test_studio_budget.py`:
a 1,000,000-byte bundle and a 2 s headless load (835 KB and 163 ms measured
on 2026-11-01).

1. **Pyodide / WASM Bridge:**
   - The minimal Pyodide distribution requires ~10–15 MB compressed (>30 MB uncompressed),
     exceeding the 1,000,000-byte bundle budget (`test_offline_bundle_fits_its_size_budget`)
     by more than 10×.
   - Headless WebAssembly initialization in Node/Chromium requires 3–5 seconds,
     exceeding the 2.0 s load budget (`test_every_script_loads_headless_within_the_startup_budget`).
2. **Codegen from Python Core:**
   - The authoritative core (`src/shared/python/putting_dynamics`) depends on
     `scipy.integrate.solve_ivp` and `numpy`. Compiling or transpiling this dependency
     graph to standalone JS without heavy numerical runtime libraries requires maintaining
     custom numerical integrators in JS—which is precisely the role served by `js/putting_physics.js`.

**Decision:**
`js/putting_physics.js` is retained with recorded offline budget compliance.
Parity is continuously enforced by:

- Automated cross-language validation in `tests/apps/putting_studio/test_js_fixtures.py`
  against `src/config/putting_reference_fixtures.json`.
- Strict zero-network and offline size gates in `tests/apps/putting_studio/test_studio_budget.py`.

## PyQt6 Parity and Claims Decision

- **Parity Exemption:** The studio's exploratory 3D vector grip/torque explorer, scenario table,
  and dispersion views are web-first visualizers exempt from PyQt6 desktop porting;
  conversely, interactive green-mesh authoring belongs exclusively to the PyQt6 tool.
  Recorded as `exempt` under `tools.putting_studio` in `src/config/feature_parity.json`.
- **Claims:** All model outputs remain under the prominent not-validated research notice
  (`validationNotice` in `index.html`). No calibrated make-rate, capture-speed, or strokes-gained
  figures are claimed until independent empirical data supports them.

## Ongoing Invariants

- Every change must keep `tests/apps/putting_studio/test_js_fixtures.py`
  green against the shared fixtures in
  `src/config/putting_reference_fixtures.json`.
- No expected value in that test file may come from this model's own
  output -- only an externally sourced fixture, or an exact
  cross-language parity check against the Python value it mirrors.
- Fixtures detect drift; they do not prove physical accuracy.
