# Complete or remove stub/partial API endpoints — web app must not present non-functional features

## Problem

Several API routes the web UI builds pages on are stubs or partial implementations, so the web app _appears_ to have parity it does not have (the same honesty problem previously hit elsewhere in the org). Inventory from this review:

| Route file                            | Endpoints                                                                    | State                                                 |
| ------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------- |
| `src/api/routes/analysis_tools.py`    | `GET /analysis/metrics`, `GET /analysis/statistics`, `POST /analysis/export` | stubs (web `AnalysisTools` page renders them as real) |
| `src/api/routes/analysis_tools.py`    | `POST /simulation/position`                                                  | partial                                               |
| `src/api/routes/character_builder.py` | character-builder endpoints                                                  | stub (web `CharacterBuilder` page exists)             |
| `src/api/routes/data_explorer.py`     | datasets list/preview/stats/import/filter                                    | partial                                               |
| `src/api/routes/model_explorer.py`    | tree inspect/compare                                                         | partial                                               |
| export pipeline                       | CSV stub; HDF5/MAT/C3D advertised in PyQt6 but absent from API               | incomplete                                            |

## Proposed fix

For each endpoint above, do exactly one of:

1. **Complete it** against the real shared implementation (preferred where the PyQt6/desktop implementation already exists — e.g. analysis statistics should come from the shared recorder/analysis services, not a canned dict); or
2. **Remove/410 it** and hide the corresponding web UI affordance until the parity issue implementing it lands (gate via the feature-parity registry, see registry issue in this epic).

Either way the web page must reflect real capability: no hardcoded sample data presented as results.

Add a CI guard: a test that walks registered routes and asserts none return the sentinel "stub" payloads (e.g. mark stubs with `X-UD-Stub: 1` header until removed, and assert the header set is empty by a target date / registry status).

## Acceptance criteria

- [ ] Every endpoint in the table is either real or gone (with web UI updated accordingly)
- [ ] No web page renders fabricated/canned data as if it were simulation output
- [ ] Feature-parity registry statuses updated to match reality

## References

- `ui/src/pages/AnalysisTools.tsx`, `ui/src/pages/CharacterBuilder.tsx`, `ui/src/pages/DataExplorer.tsx`
- Related root-cause pattern: parity-only tests don't catch stubbed behavior (#7407-style gap)
