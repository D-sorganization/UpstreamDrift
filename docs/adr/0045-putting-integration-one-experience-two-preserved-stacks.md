# ADR-0045: Putting Integration — One Experience, Two Preserved Physics Stacks

- Status: Proposed
- Date: 2026-08-30
- Decision Makers: repo owner (approval required before any implementation)
- Related Issues/PRs: #9143, Tools#4800 (P2/P9), Tools#4816, Tools#4819, launcher tiles `putting_green` and `rate_of_closure`

## Context

UpstreamDrift now carries two complete putting simulators, and a user who
putts in both gets different answers with no explanation:

| Capability    | UD `putting_green` engine                                                                | Tools putting (vendored, Impact Explorer tab)                                                                        |
| ------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| Surface model | Topography presets, grid `_surface_io` JSON, `.npy`/GeoTIFF loaders, scattered-point RBF | Parametric heightfield + grid wire `swing_sim.green_surface/1` (bilinear)                                            |
| Roll physics  | `ball_roll_physics` — µ ≈ `0.196/stimp` × height-of-cut/condition/grain factors          | `putting/roll.py` — µ ≈ `0.559/stimp` (USGA stimpmeter geometry, 1.83 m/s release)                                   |
| Hole capture  | Radius test + 1.5 m/s lip-out heuristic                                                  | Holmes (1991)/Penner (2002) speed-dependent effective radius                                                         |
| Impact/stroke | None — launch state is an input                                                          | Full 2-D impact solve (aim/face/path/attack/offsets), mesh-derived putter MOI, stroke import from engines            |
| Extras        | Practice mode, wind physics, checkpoints, API route, web page + `puttingPlayback.ts`     | Dispersion Monte Carlo, putter-fitting counterfactuals, `putting_result/2` wire, 3D playback on the shared transport |

The two stacks were already bridged at the data level by Tools#4819 (P9): a
runtime-free adapter converts UD topographies to/from the
`swing_sim.green_surface/1` wire, refuses the non-conservative weighted-slope
field rather than approximating it, and **gate-pins the roll-out divergence at
a constant ratio ≈ 2.854** (the two µ-laws share the `1/stimp` form but assume
different stimpmeter release speeds). The divergence is physics, not a bug —
each model is internally consistent.

Constraint set by the owner: **no functionality may be deleted or limited.**
Consolidation means integration, not deprecation.

## Decision

Adopt a layered ownership model in which each capability has exactly one home
and every surface reaches it through the existing wires:

1. **Green authoring and terrain stay with UD `putting_green`.** Its surface
   presets, importers (grid/`.npy`/GeoTIFF/RBF), practice mode, and wind
   remain the authoring experience. Nothing moves out of it.
2. **Stroke, impact, and analytics stay with Tools putting.** Impact solve,
   putter MOI, dispersion, counterfactuals, `putting_result/2`, and 3D
   playback remain the analytics authority. Nothing moves out of it.
3. **The P9 wire is the only bridge.** A green authored in `putting_green`
   (any importer) exports through the adapter into the Tools simulator; a
   `green_surface/1` document loads into `putting_green`. Neither stack
   imports the other's runtime.
4. **The physics divergence becomes a user-visible, named choice instead of a
   silent fork.** Both roll models are preserved and exposed as named,
   provenance-carrying options — `ud-legacy-roll/1` (agronomic factors:
   height-of-cut, condition, grain) and `usga-stimp-roll/1` (stimpmeter
   geometry + Holmes/Penner capture) — selectable wherever a putt is
   integrated, with the active model recorded in every result document.
   Results from different models are never numerically compared without the
   model names attached. Neither model is removed; the ≈2.854 gate keeps the
   documented relationship honest.
5. **Launcher/tile presentation:** both tiles remain. `putting_green`'s
   description gains "green authoring, practice" framing; the Impact
   Explorer's putting tab gains an "Import green from Putting Green…" action
   (the adapter already exists). A later phase may add cross-links between
   the two surfaces; neither tile is removed.

## Alternatives Considered

1. **Retire UD `putting_green`, keep only Tools putting.** Rejected: deletes
   practice mode, wind, GeoTIFF/RBF import, and the web page — violates the
   no-deletion constraint, and the authoring UX is genuinely good.
2. **Retire Tools putting tab inside UD, keep `putting_green`.** Rejected:
   deletes the impact solve, MOI fitting, dispersion, counterfactuals — the
   scientific core the epics just built.
3. **Force one µ-law.** Rejected for now: both models are defensible under
   different assumptions; silently rewriting one would change every archived
   result. Naming the models preserves both and makes the choice explicit.
   A future calibration study may promote one to default — that is a
   physics decision with its own evidence, not an architecture decision.
4. **Merge the codebases.** Rejected: cross-repo runtime imports invert the
   Tools-is-a-leaf dependency rule; the wire boundary is the architecture.

## Consequences

- Positive: zero capability loss; one bridge to maintain (already tested);
  divergence becomes informative instead of confusing; each stack keeps its
  release cadence.
- Negative: two roll models remain to maintain; the model-selection surface
  adds one concept for users ("which physics?") — mitigated by a sensible
  default per surface (authoring surfaces default to `ud-legacy-roll/1`,
  analytics surfaces to `usga-stimp-roll/1`, both switchable).
- Follow-ups (each its own issue, sized after approval):
  F1 model-name plumbing into `putting_green` results; F2 "Import green"
  action in the Impact Explorer putting tab (both runtimes); F3 cross-links
  between the two tiles; F4 the UD-side consumer test from #9143.

## Validation

- The existing P9 cross-engine gates (ratio pin, sign/monotonicity parity)
  stay mandatory; a change in either µ-law fails them.
- New gate with F1: every putt result document names its roll model; a
  result without a model name is refused by the fail-closed readers.
- No test that exists today is deleted or weakened by this ADR.
