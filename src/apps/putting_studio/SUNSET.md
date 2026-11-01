# Sunset Criterion: `js/putting_physics.js`

`js/putting_physics.js` is a **temporary**, hand-kept copy of putting
physics. The single source of truth is the Python core at
`src/shared/python/putting_dynamics/`; the PyQt6 tool and this HTML
studio are both clients of it, not peers. The copy exists only because
the studio opens from `file://` with no build step and no runtime CDN
dependencies, so it cannot `import` the Python package directly.

## Sunset Criterion

This copy is replaced by a generated or bridged core (Pyodide/WASM, or
codegen from the Python source) once that core passes a measured
offline asset-size and startup budget, to be set in the final studio slice, 8l (epic
#14141).

## Until Then

- Every change must keep `tests/apps/putting_studio/test_js_fixtures.py`
  green against the shared fixtures in
  `src/config/putting_reference_fixtures.json`.
- No expected value in that test file may come from this model's own
  output -- only an externally sourced fixture, or an exact
  cross-language parity check against the Python value it mirrors.
- Fixtures detect drift; they do not prove physical accuracy.
