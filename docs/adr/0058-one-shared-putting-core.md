# ADR-0058: One Shared Putting Physics Core

- Status: Accepted
- Date: 2026-11-01
- Decision Makers: repo owner (design of record for epic #14141, approved 2026-10-31)
- Related Issues/PRs: #14141, supersedes decision 4 of [ADR-0045](0045-putting-integration-one-experience-two-preserved-stacks.md)

## Context

ADR-0045 kept two putting roll models as named, selectable options:
`ud-legacy-roll/1` (µ ≈ 0.196/stimp, in the UD `putting_green` engine) and
`usga-stimp-roll/1` (µ from the USGA Stimpmeter geometry). It also rejected
"force one µ-law" until evidence supported a choice.

Epic #14141 made that choice. Its design of record, approved by the owner on
2026-10-31, names `src/shared/python/putting_dynamics/` the single source of
truth for putting physics, with every client (PyQt6 tool, API, HTML studio,
MCP tools, strokes-gained model) built on it. Steps 4–10a carried it out:

- The shared turf model applies the USGA law with the UD agronomic inputs
  (grass type, condition and grain), so those inputs survive.
- The API routes (7a, 7c), shared green reading (7b), PyQt6 scene (7d), MCP
  tools (9a) and engine loader (10a) all run `usga-stimp-roll/1`. So does
  the strokes-gained putting provider from #14177 (`CurvePuttingProvider`
  over `putting_analytics`, parity-tested against the shared solver), which
  delivered step 9's make probability; the planned step 9b duplicate of it
  was dropped.
- Every result still names its roll model, as ADR-0045 F1 required.

## Decision

1. **One roll model is active: `usga-stimp-roll/1`.** `ud-legacy-roll/1` is
   retired. No user surface runs it, and it must not return to the shared
   package (`tests/unit/putting_dynamics/test_turf.py`). This supersedes
   ADR-0045 decision 4 and its default-per-surface rule.
2. **Archived results stay interpretable.** A result document carries its
   roll-model name, so a `ud-legacy-roll/1` result is never compared with a
   `usga-stimp-roll/1` one without that label.
3. **The rest of ADR-0045 stands.** Green authoring and terrain import stay
   with UD (decision 1). Tools putting stays the impact and analytics
   authority, and the `green_surface/1` wire stays the only bridge
   (decisions 2 and 3). Both launcher tiles stay (decision 5).
4. **No unique feature is deleted.** The owner rule in ADR-0045 ("no
   functionality may be deleted or limited") still applies. Step 10 removes
   duplicate physics only. A feature that exists only in the legacy engine
   is moved onto the shared core before its legacy copy is deleted, or kept.
   The wind model is the recorded exception: wind > 0 has been refused at
   the API since #14141 step 7a, as a documented non-port.

## Alternatives Considered

1. **Keep both roll models selectable (ADR-0045 as written).** Rejected by
   the design of record: two cores drift, and every client would need both.
2. **Promote `ud-legacy-roll/1` instead.** Rejected: its 0.196 coefficient
   has no outside derivation, while the USGA law follows from the
   Stimpmeter's published geometry.

## Consequences

- Positive: one physics core to test against the shared reference fixtures.
  The PyQt6 tool, API and agent tools agree by construction; the studio's
  temporary JS copy is held to the same fixtures until its sunset.
- Negative: at equal stimp and launch speed, putts roll out about 2.854
  times shorter than under `ud-legacy-roll/1` (the µ ratio 0.559/0.196 pinned
  by Tools#4819), so new results differ from archived legacy ones. The
  roll-model name on every result marks this.
- Fixtures show drift, not accuracy. The shared law is not a calibration
  claim, and the turf factors remain provisional until outside measurements
  support them.

## Validation

- `tests/unit/putting_dynamics/test_turf.py` keeps the legacy coefficient
  out of the shared package.
- The route, GUI, MCP and engine tests assert `usga-stimp-roll/1` on their
  results.
- The Tools P9 cross-stack gates stay mandatory for the `green_surface/1`
  wire.
