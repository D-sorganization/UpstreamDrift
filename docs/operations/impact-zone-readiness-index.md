# Impact Zone Readiness Index

<!-- AUTO-GENERATED — do not edit by hand. -->
<!-- Regenerate with: python3 -m scripts.generate_industrial_readiness_index -->

Generated from [`src/config/impact_zone_readiness.json`](../../src/config/impact_zone_readiness.json) (ledger v1.0.0).

Execution index for epic [#9546](https://github.com/D-sorganization/UpstreamDrift/issues/9546), the 2026-09-04
review's Impact Zone and Impact Explorer product program. Tools owns the
shared impact/flight runtime, so provider fixes land there first and this
record says which reviewed pins UpstreamDrift has consumed, what proves
it, and which product slices remain open. It is a software-correctness
record only — scientific and human
qualification are recorded separately, through the design-manual
governance pathway.

- **Release status:** 🔴 blocked
- **Reconciled against:** `5347cba0f4378cd72a6e8afea9fb27c8bfe5db75` on 2026-09-18
- **Audit snapshot:** `1f69a51fce997932f04a6ad1dd95bf4d065ba971` (context, not current branch identity)
- **Queue:** 1 merged · 5 open

## Priority Implementation Queue

- **I1** · [#9547](https://github.com/D-sorganization/UpstreamDrift/issues/9547) · P1 · 🔴 open · owner unassigned, depends on [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)
- **I2** · [#9548](https://github.com/D-sorganization/UpstreamDrift/issues/9548) · P1 · 🔴 open · owner unassigned, depends on [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)
- **I3** · [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549) · P1 · 🔴 open · owner unassigned, depends on [#9349](https://github.com/D-sorganization/UpstreamDrift/issues/9349)
- **I4** · [#9550](https://github.com/D-sorganization/UpstreamDrift/issues/9550) · P2 · 🔴 open · owner unassigned, depends on [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)
- **R1** · [#9484](https://github.com/D-sorganization/UpstreamDrift/issues/9484) · P1 · ✅ merged
- **R2** · [#9349](https://github.com/D-sorganization/UpstreamDrift/issues/9349) · P2 · 🔴 open · owner Tools provider (rate_of_closure Impact Explorer tab), depends on —

### I1 — Impact: Reject Unfinished Contact Before Exporting a Post-Impact State

🔴 open · P1 · [#9547](https://github.com/D-sorganization/UpstreamDrift/issues/9547)

- **Owner:** unassigned
- **Depends on:** [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)
- **Source:** —
- **Existing tests:** `tests/shared_contracts/test_impact_interval_provider.py`

**Narrow PR plan.** Provider half landed: Tools #5088 (`183b4bb1fe34ec408c5cd400bb553bea999a88a0`, 2026-09-08) adds `ImpactTermination` (SEPARATED / TIME_LIMIT / NO_CONTACT), `ImpactIntervalResult.contact_completed`, a floor(T/dt) step budget that no longer runs a step past the configured cap, and `IncompleteContactError` carrying the partial trace from `to_post_impact_state()`. The audit pin `3d93bb2c89813e17551814d3be7e895f791e29af` had none of these names. UpstreamDrift consumes it through the vendored pin `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1` (first carried by `c487265f1ebc9c61a2e124267ad9cd1c96a6c007` #9916, current bump `e9b3278a13b497cfa5363b2a707494d9d5695c77` #10373); `tests/shared_contracts/test_impact_interval_provider.py` drives the audit probe (1e-5 s cap, ball still compressed under >10 kN) through the provider and pins TIME_LIMIT, the hard cap, the refusal and the inspectable partial trace. Still open on the product side: session/export schemas and the workbench UI do not yet carry a completion status because the live interval seam (I3, #9549) does not exist; land it there, then re-verify this entry's acceptance boxes against the seam's export schema before recording a merge.

### I2 — Impact: Compute Contact Energy Loss Independently Instead of Zeroing the Audit Residual

🔴 open · P1 · [#9548](https://github.com/D-sorganization/UpstreamDrift/issues/9548)

- **Owner:** unassigned
- **Depends on:** [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)
- **Source:** —
- **Existing tests:** `tests/shared_contracts/test_impact_interval_provider.py`

**Narrow PR plan.** Provider half landed: Tools #5079 (`669b478e1a8f93e8dbbed0aa3c6a7d263b011229`, 2026-09-07) reports recoverable spring energy as `stored_contact_energy_{initial,final}_j`, accumulates `unilateral_release_energy_j` only at identified tensile-clip steps, integrates dashpot/friction and torsional-grip damping independently, and keeps `energy_residual_j` as the signed remainder of the declared ledger. The audit pin still had `unilateral_release = max(0.0, raw_energy_residual)`, which relabelled the 5.777 J deficit of the truncated probe as release with a 0.0 J residual. Consumed through the same vendored pin as I1; the UpstreamDrift consumer contract asserts zero release, positive stored energy and the residual identity on the truncated probe. Still open on the product side: the audit terms and their dt-convergence limits are not surfaced in any UpstreamDrift report or UI, and no qualified output is blocked on a failed audit, because no live run consumes the interval solver yet (I3, #9549). Surface the audit in the I3 run record and export, then re-verify this entry's acceptance boxes before recording a merge.

### I3 — Impact: Integrate the Contact-Interval Solver Into Live Runs, Playback and Export

🔴 open · P1 · [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)

- **Owner:** unassigned
- **Depends on:** [#9349](https://github.com/D-sorganization/UpstreamDrift/issues/9349)
- **Source:** `ui/src/pages/ImpactExplorer.tsx`
- **Existing tests:** `ui/src/pages/ImpactExplorer.test.tsx`

**Narrow PR plan.** Confirmed still open at the reconciliation SHA and the vendored pin: nothing under `vendor/ud-tools/src/rate_of_closure/{simulation,application,ui,web}` imports `swing_sim.impact_interval`, so the workbench has no selectable live interval model, no `SimulationRun` record carrying termination/audit, and no export of the interval trace; the existing imported 3-D playback (#9353) is not that seam. Ownership: the run record, model selection and presenter seam are Tools work under Tools #4130/#4946 and must land there first with a migration-compatible default for existing instantaneous models; UpstreamDrift then consumes the reviewed pin, extends the `/tools/impact-explorer` route and desktop tile only if the seam changes their contract, and adds consumer-contract tests for the run record's completion state and audit. Narrow PR order: (1) Tools run record + model selection with the interval solver behind an explicit option; (2) Tools presenter/export carrying `ImpactTermination` and `ImpactIntervalAudit`; (3) UpstreamDrift pin bump plus consumer contracts, closing the I1/I2 product boxes in the same slice.

### I4 — Impact: Qualify the Installed Product, Physics Claims and Desktop/Web Experience

🔴 open · P2 · [#9550](https://github.com/D-sorganization/UpstreamDrift/issues/9550)

- **Owner:** unassigned
- **Depends on:** [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)
- **Source:** —
- **Existing tests:** —

**Narrow PR plan.** Cannot start before I3: end-to-end qualification of interactive and saved playback needs a live interval run to qualify. When I3 lands, this slice owns dimensional/sign/frame, energy/momentum and timestep-limit tests at the product boundary, desktop/web golden contract parity, calibration and independent physical validation for any advertised accuracy, rendered usability/accessibility inspection, and clean install/package evidence. Approximation modes named in Tools #4251 (gear-effect ramp, COR(v), spring-damper spin, tensor defaults, D-plane/energy presentation) stay labelled and are rechecked at implementation head rather than assumed. No entry here may be marked merged on a mock-only run, a skipped hardware test or a stale screenshot.

### R1 — The Impact Explorer Web Route Has No Build Step in CI

✅ merged · P1 · [#9484](https://github.com/D-sorganization/UpstreamDrift/issues/9484)

- **Merge SHA:** `ba841ae5246d26ef762deb0b9d0c0ca13ef1b784`
- **Implementation:** `.github/workflows/ci-standard.yml` · `scripts/check_declared_route_producers.py` · `ui/src/pages/ImpactExplorer.tsx`
- **Tests:** `tests/scripts/test_declared_route_producers.py` · `ui/src/pages/ImpactExplorer.test.tsx`

**Acceptance evidence.** CI Standard's `impact-explorer-web-build` job builds `vendor/ud-tools/src/rate_of_closure/web/dist` from the pinned Tools tree with the route fallback's own command (`npm ci`, `npm run build -- --base=/impact-explorer-app/`, node 22 per Tools' `rate-of-closure-web-distribution.yml` authority), verifies the base path and joins the required `quality-gate` aggregate, so a clean checkout that passes CI serves the real app at `/tools/impact-explorer` rather than the fallback; `scripts/check_declared_route_producers.py` fails any launcher tile that declares a web route no pipeline produces. Shipping the built bundle inside the wheel or image is not part of this entry; it remains the open artifact decision under #9417.

### R2 — ADR-0046 G2: Re-Point the Two Workbenches at the Canonical Layer, Module by Module

🔴 open · P2 · [#9349](https://github.com/D-sorganization/UpstreamDrift/issues/9349)

- **Owner:** Tools provider (rate_of_closure Impact Explorer tab)
- **Depends on:** —
- **Source:** `src/tools/launch_monitor_model/__init__.py` · `src/tools/launch_monitor_analytics/gui.py`
- **Existing tests:** `tests/unit/launch_monitor/test_canonical_layer_parity.py`

**Narrow PR plan.** Half done. The UpstreamDrift 9-tab workbench is re-pointed: ADR-0046 Stage 2 waves 1-3b retired all 28 port-up/merge modules onto the canonical `shared.python.launch_monitor` layer, and `src/tools/launch_monitor_model/` now holds only the re-export facade, the app-local `project`, `launch_monitor_data` and `strokes_gained_baseline`. The other workbench is not: at the vendored pin `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1`, `vendor/ud-tools/src/rate_of_closure/` still carries its private `launch_monitor_performance`, `launch_monitor_analysis`, `launch_monitor_longitudinal` and sibling modules, and only `launch_monitor_strokes_gained.py` imports the canonical layer. Remaining slices are Tools PRs, one per Impact Explorer tab, each retiring a private module only after that tab's tests pass against the canonical layer; UpstreamDrift consumes each reviewed pin and re-runs the 71 ADR-0046 G0 drift gates, which stay in place until the last private copy is gone.

## Acceptance Criteria

### 🟡 Partial — I1 / I2 Interval Completion and Independent Energy Accounting Are Fixed in Tools and Their Reviewed Immutable Pin Is Consumed Here.

- **Blockers:** [#9547](https://github.com/D-sorganization/UpstreamDrift/issues/9547), [#9548](https://github.com/D-sorganization/UpstreamDrift/issues/9548), [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)

Tools #5088 and #5079 are in the vendored pin `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1` and `tests/shared_contracts/test_impact_interval_provider.py` exercises both through the provider on the audit probe (3 passed, `--tools-mode=vendored`, Windows / Python 3.13). The product-side boxes of both children (completion status in schemas and UI, audit surfaced in reports, qualified output blocked on a failed audit) wait on the live seam.

### 🔴 Unmet — I3 Live Interval Selection, Run Record and Presenter Seam Exist With a Migration-Compatible Default.

- **Blockers:** [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549), [#9349](https://github.com/D-sorganization/UpstreamDrift/issues/9349)

No module under `vendor/ud-tools/src/rate_of_closure` imports the interval solver at the pin; there is no selectable interval model, run record or export. Tools #4130/#4946 own the seam.

### 🟡 Partial — A Clean Documented Install Opens the Real App, Not the Fallback.

- **Blockers:** [#9550](https://github.com/D-sorganization/UpstreamDrift/issues/9550)

CI builds the Impact Explorer bundle from the pinned tree and gates declared route producers (#9484, merged). The bundle is not yet shipped inside any installable artifact, so a clean install from the wheel or image still serves the documented fallback; that is the U4 (#9417) artifact decision, tracked in the industrial-readiness ledger.

### 🟡 Partial — No-Hit, Unfinished Contact and Unavailable Models Are Distinct States End to End.

- **Blockers:** [#9547](https://github.com/D-sorganization/UpstreamDrift/issues/9547), [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)

The provider distinguishes NO_CONTACT, TIME_LIMIT and SEPARATED and refuses to export the unfinished case; the consumer contract pins all three. The states are not yet distinct in any session schema, export or UI because no live run consumes the solver.

### 🔴 Unmet — Release Requires Dimensional/Sign/Frame, Energy/Momentum and Timestep-Limit Tests; Desktop/Web Golden Parity; Calibration and Independent Physical Validation; Rendered Usability Inspection; Clean Install Evidence.

- **Blockers:** [#9550](https://github.com/D-sorganization/UpstreamDrift/issues/9550), [#9549](https://github.com/D-sorganization/UpstreamDrift/issues/9549)

Nothing beyond the provider-level solver tests and this consumer contract exists yet. I4 owns the product-boundary qualification and cannot start before I3.

### 🟡 Partial — Approximation Modes Remain Labelled and Cannot Imply Validated Prediction.

- **Blockers:** [#9550](https://github.com/D-sorganization/UpstreamDrift/issues/9550)

The vendored shaft provider contract (`tests/shared_contracts/test_impact_shaft_provider.py`) refuses a promoted `validation_status`, and the design-manual governance pathway remains the only scientific authority. The Tools #4251 approximation list (gear-effect ramp, COR(v), spring-damper spin, tensor defaults, D-plane/energy presentation) has not been rechecked at implementation head; I4 does that.

### ✅ Met — Software Correctness, Numerical Validity and Human Qualification Remain Separately Recorded.

- **Blockers:** —

This ledger records software correctness only, under the same contract as the industrial-readiness ledger. Scientific qualification stays in the design-manual governance pathway (`scripts/config/design_manual_governance.json`, `python3 -m scripts.check_design_manual_governance`); a passing entry here is not scientific or human approval.

## Keeping This Record Honest

`src/config/industrial_readiness_loader.py` refuses a ledger that claims
more than the tree supports. A merged entry must carry a 40-character
merge SHA, at least one test path, and user-visible acceptance evidence;
an open entry must carry no merge SHA, an owner, a plan, and a dependency
when no owner is named. Every referenced path must exist. Every open
issue must appear in an acceptance blocker list, and the release status
cannot read `ready` while any entry is open or any criterion is unmet.

An issue closure, a mock-only success, a changed golden file or a raised
tolerance is not evidence of correctness and must not be recorded here.
