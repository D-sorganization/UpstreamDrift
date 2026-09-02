# ADR-0046: Launch-Monitor Analytics — Two Workbenches, One Model Layer

- Status: Accepted
- Date: 2026-08-30
- Decision Makers: repo owner (accepted 2026-08-30)
- Related Issues/PRs: launcher tiles `launch_monitor_analytics` and `rate_of_closure`; Tools launch-monitor epic (#4583 line); UD `src/shared/python/launch_monitor/`

## Context

This is the deepest duplication in the platform: two **independent,
full-depth launch-monitor analytics stacks**, developed in parallel, with no
shared code and no shared wire.

|                      | UD stack                                                                                                                                                                                                                                                                                                                                                                                               | Tools stack (vendored)                                                                                                                                                                                                                     |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model layer          | `src/shared/python/launch_monitor/` — **30 modules**: corpus, importer, schema, treatment, relationships, multivariate, modeling, comparison, dispersion, trends, longitudinal(+statistics/types), strokes_gained(+types), player_covariation(+core/types), profiles, project, outcome_proxy, conformance_bundle, dataset_reference(+contract/operations/verification), flexible_analysis, contract_v2 | `rate_of_closure/launch_monitor_*` (~12 Python modules + full TS twins): analysis(+statistics/types), canonical_v2 (+pinned golden), import, linked_scatter, longitudinal, numeric, performance, private_corpus, strokes_gained(+baseline) |
| GUI                  | Standalone 9-tab workbench (`src/tools/launch_monitor_analytics`, 1.7k LOC): Sessions, Data Treatment, Relationships, Flexible Analysis, Models, Monitor Comparison, Dispersion, Trends, Reports                                                                                                                                                                                                       | Launch Monitor Analytics tab inside the Impact Explorer, in **both** runtimes (PyQt + React), with source-backed strokes-gained v2, dispersion, session trends, longitudinal inference                                                     |
| Distinctives         | Monitor **comparison**, data **treatment** pipeline, flexible analysis, player covariation, dataset-reference verification                                                                                                                                                                                                                                                                             | **Cross-runtime parity with pinned goldens**, canonical v2 wire, explicit-identity projects, private-corpus boundary (no private rows in project files), linked scatter into the impact model's context                                    |
| Verified independent | `strokes_gained.py` heads differ in structure, imports, and helpers; Tools has **no** `shared/python/launch_monitor` package; UD's tab imports only its own layer                                                                                                                                                                                                                                      |                                                                                                                                                                                                                                            |

Both are good. Neither is a subset of the other. The duplication cost is
real: a metric defined twice (strokes gained, dispersion) can silently
diverge exactly like the putting µ-laws did, but here there is _no adapter
and no gate_ — nothing would even reveal it.

Constraint set by the owner: **no functionality may be deleted or limited.**

## Decision

Converge on **one model layer, two preserved workbenches**, in three
non-destructive stages. Both GUIs survive throughout; users lose nothing at
any stage.

1. **Stage 0 — the drift gate (immediately, before any migration).**
   Define the overlap set (strokes gained, dispersion, longitudinal trend
   statistics) and add cross-stack consistency gates: identical synthetic
   session in → numerically compared out, with any _legitimate_
   methodological difference documented and pinned, exactly as the putting
   ratio gate does. This makes the current divergence measurable before
   anything moves.
2. **Stage 1 — canonical model layer in Tools.** Tools is the fleet's DRY
   leaf (UD vendors it; Gasification_Model consumes it), so the canonical
   layer grows there, in shared code (not inside `rate_of_closure`).
   Capabilities that exist only in UD's layer (comparison, treatment,
   player covariation, dataset-reference verification, flexible analysis)
   are **ported up** to the canonical layer with their tests — ported, not
   reimplemented; UD's implementations are the reference and its authors'
   attribution is retained. Capabilities that exist only in Tools' layer
   (canonical v2 wire, private-corpus boundary, cross-runtime goldens)
   are already home.
3. **Stage 2 — both workbenches consume the canonical layer.** The UD 9-tab
   workbench keeps its exact UI and tab set, re-pointed module-by-module at
   the vendored canonical layer, retiring its private copy only when each
   module's consumers are on the canonical one **and its tests pass against
   it**. The Impact Explorer tab likewise. The UD workbench remains the
   "deep desk" surface; the Impact Explorer tab remains the in-context
   surface; the launcher keeps both tiles with descriptions that state the
   relationship ("the same analytics engine").

## Alternatives Considered

1. **Retire the UD workbench.** Rejected: deletes monitor comparison, data
   treatment, flexible analysis, player covariation — capabilities the
   Tools tab does not have; violates the constraint.
2. **Retire the Tools tab.** Rejected: deletes cross-runtime web analytics
   entirely (the UD workbench is desktop-only), plus the private-corpus
   boundary and pinned parity.
3. **Keep both stacks, add only gates (stop at Stage 0).** Viable minimum,
   and Stage 0 is valuable alone — but it locks in double maintenance of
   30+12 modules forever. Kept as the fallback if Stage 1 stalls.
4. **Canonical layer in UD instead of Tools.** Rejected: inverts the fleet
   dependency direction; Gasification_Model and the web twins could not
   reach it.

## Consequences

- Positive: one definition of every metric; UD-only capabilities become
  available to the web runtime for the first time (they arrive in the
  canonical layer's TS twins); both UIs keep their identity.
- Negative: Stage 1 is real porting work with review load; during the
  transition the drift gates are the safety net; vendored-pin bumps become
  load-bearing for the UD workbench.
- Follow-ups (own issues, sized after approval): G0 gates (#9354, merged);
  G1 port plan per module — the reviewed inventory of the 30 UD modules with
  keep-port-already-home classification is
  [ADR-0046 G1: Launch-Monitor Port Plan](0048-launch-monitor-port-plan.md) (#9348);
  G2 re-pointing PRs per tab.

## Validation

- Stage 0 gates run in both repos' CI from day one.
- Every Stage 2 re-point PR must show the tab's existing tests passing
  against the canonical layer before the private module is removed —
  removal of a private module with failing or skipped tab tests is the
  explicit non-goal.
- The launcher-manifest parity contract keeps both tiles' claims honest.

## Amendment 1 (2026-08-30) — G0/G1 Evidence Corrections

Measurement corrected two factual claims in this record; the decision stands,
the inventory it feeds must not repeat them:

1. **`flexible_analysis` and `player_covariation` are NOT UD-only.** Tools
   carries same-shaped counterparts (six identically named frozen dataclasses
   in the flexible-analysis pair; the same three-module within-player +
   Fisher-z design in covariation, 1,098 vs 570 lines). Neither side is a
   superset. Both pairs need the G0 treatment — measured comparison on the
   shared fixture (G0.1) — before classification.
2. **The taxonomy needs a merge bucket.** `corpus.py` (UD) and
   `launch_monitor_private_corpus.py` (Tools) share the same env var and
   parquet path with complementary, non-overlapping guarantees; the correct
   outcome is a merged module, which port-up/already-home/app-local cannot
   express.

The authoritative inventory, decisions, and port order live in
[0048-launch-monitor-port-plan.md](0048-launch-monitor-port-plan.md) (G1, #9348), which supersedes
this record's capability lists where they conflict.
