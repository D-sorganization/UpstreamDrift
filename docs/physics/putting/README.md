# Putting Studio v4 (Gemini Suite) — Panel Documentation

Epic **#14141**: integrate the Gemini-built **3D Counterbalanced Putting Dynamics & Undulating Green Roll Suite (v4.0)** as a flagship UpstreamDrift feature. **Production wiring is deferred** until the expert integration panel publishes target architecture.

## Reference source (not production)

| Item | Location |
| --- | --- |
| Untouched code, tests, HTML, monograph QMD | Branch `ref/putting-v4-gemini-source` → `vendor_review/putting_v4_gemini/` |
| Mathematical report (PDF) | Not in git (no LFS); retain from handoff / Google Drive |
| Panel turnover & manifest | This directory (`TURNOVER_*.md`, `HANDOFF_*.md`) |

## Goals

1. **Professional grade** — TDD, design-by-contract, law-of-demeter, and DRY review before any merge from `vendor_review/` into `src/`.
2. **Parity** — aligned capabilities between the standalone HTML studio and `src/tools/putting_green_gui/` (PyQt6).
3. **Standalone HTML** — keep a shareable studio artifact, not only an embedded view.
4. **Panel-gated design** — module boundaries, deprecations, MCP registration, and launcher/feature-parity updates follow panel ADR.

## Known handoff gaps

- Reference pytest: **3 / 49** failures (bare-name imports; turf path assumption).
- `putting_dynamics_v3.py` ≈ **1,177 LOC** (exceeds default 500 LOC/file until split or budget exception).
- Fleet does **not** adopt handoff “zero MuJoCo references”; engine integrations may cite MuJoCo where required.

## Module overlap map (Gemini → existing UD → panel target)

| Gemini module (v4) | Overlaps today | Intended target (panel) |
| --- | --- | --- |
| `putting_dynamics_v3.py` | `src/shared/python/putting_dynamics/` (Epic #8345) | Unified putting engine facade under shared physics; split to meet LOC budgets |
| `turf_properties.py` | `src/engines/physics_engines/putting_green/python/turf_properties.py`, putting_dynamics friction/collision | Single turf/cup-capture spec consumed by engine + shared layers |
| `grip_nullspace_stabilization.py` | (no full equivalent; partial biomechanics elsewhere) | Shared grasp / null-space preload module for studio + sim |
| `green_surface_undulation.py` | `putting_dynamics/surfaces.py`, putting green engine | Shared undulating green geometry + gradients |
| `strokes_gained_analytics.py` | `src/shared/python/sg_optimizer/`, `src/tools/sg_optimizer/` | Player Bellman SG, batch Monte Carlo, variance channels |
| `stimpmeter_reader.py` | Stimpmeter logic scattered in green GUI / engine | Shared virtual stimpmeter + green-read solver |
| `putting_mcp_server.py` | `src/sim_run/mcp.py`, sim MCP registry | Registered putting MCP tools for agents + studio export |
| `putting_3d_torque_studio_v3_2.html` | `src/tools/putting_green_gui/` (PyQt6) | `src/apps/putting_studio/` (or panel path) + parity matrix vs PyQt6 |
| Test suites under `tests/unit/physics/putting/` | Existing putting / green tests (sparse vs v4 breadth) | Relocated imports; CI architecture suite aligned with repo helpers |

## Related documents in this folder

- [`TURNOVER_DOC_PUTTING_3D_DYNAMICS_v3_2.md`](TURNOVER_DOC_PUTTING_3D_DYNAMICS_v3_2.md) — layout, MCP usage, verification commands.
- [`HANDOFF_PROMPT_AND_MANIFEST_v3_2.md`](HANDOFF_PROMPT_AND_MANIFEST_v3_2.md) — feature inventory and verification status from handoff.
