# Matched Swing Program Wave Structure and Roadmap

**Author:** Dieter Olson (`agent:local`)  
**Date:** 2026-09-17  
**Issue:** MS-06 (#10327), Epic #10363

---

## 1. Wave Overview

The Matched Swing Program is organized into six structured execution waves, ensuring fail-closed contracts, verified evidence, and multi-agent coordination precede broad release qualification.

```
Wave 0 (Foundations) ──► Wave 1 (Kinematics) ──► Wave 2 (Dynamics) ──► Wave 3 (Cross-Engine) ──► Wave 4 (UI/UX) ──► Wave 5 (Release)
```

---

## 2. Detailed Wave Breakdown

### Wave 0: Program Foundations, Hygiene & Ledger (MS-01 – MS-07, MS-50, MS-84, MS-95)

_Objective:_ Establish single sources of truth, immutable run ledger, physical acceptance contracts, dependency gates, and backlog hygiene.

| Task      | Issue                                                                   | Title                                               | Status                              |
| --------- | ----------------------------------------------------------------------- | --------------------------------------------------- | ----------------------------------- |
| **MS-01** | [#10322](https://github.com/D-sorganization/UpstreamDrift/issues/10322) | Physical acceptance contract and gates (G1/G2/G3)   | **MERGED** (`cde6d767e`, PR #10368) |
| **MS-02** | [#10323](https://github.com/D-sorganization/UpstreamDrift/issues/10323) | Matched-swing run ledger and evidence scanner       | **PR OPEN** (PR #10387)             |
| **MS-03** | [#10324](https://github.com/D-sorganization/UpstreamDrift/issues/10324) | Reconcile headline numbers with primary receipts    | **PR OPEN** (PR #10389)             |
| **MS-04** | [#10325](https://github.com/D-sorganization/UpstreamDrift/issues/10325) | Hash-lock target capture copies and marker validity | **MERGED** (`26ccd1eac`, PR #10369) |
| **MS-05** | [#10326](https://github.com/D-sorganization/UpstreamDrift/issues/10326) | Cross-engine leaderboard orchestrator module paths  | **MERGED** (`42eda4b66`, PR #10370) |
| **MS-06** | [#10327](https://github.com/D-sorganization/UpstreamDrift/issues/10327) | Program tracker doc, generator script, and banners  | **ACTIVE** (Lease held by `local`)  |
| **MS-07** | [#10328](https://github.com/D-sorganization/UpstreamDrift/issues/10328) | Backlog hygiene: adopt and label legacy issues      | **CLOSED & COMPLETED**              |
| **MS-50** | [#10343](https://github.com/D-sorganization/UpstreamDrift/issues/10343) | Make MyoSuite provider fail-closed and experimental | **MERGED** (`98ddcf8ba`, PR #10372) |
| **MS-84** | [#10357](https://github.com/D-sorganization/UpstreamDrift/issues/10357) | Shadow Tracker tile corrective (review & viewport)  | **MERGED** (`bfcb418a5`, PR #10364) |
| **MS-95** | [#10362](https://github.com/D-sorganization/UpstreamDrift/issues/10362) | Tools dependency gate and pin bump                  | **PR OPEN** (PR #10373)             |

---

### Wave 1: Data Contracts and Cross-Engine Kinematic Parity (MS-10 – MS-15, MS-20, MS-21)

_Objective:_ Common C3D marker interface, canonical tour capture normalization, and sub-millimeter static pose parity across all engines.

- **MS-10 (#10329):** Single source of truth C3D ingestion adapter.
- **MS-11 (#10330):** Tour-average marker coordinate normalization.
- **MS-12 (#10331):** Anthropometric segment table validator.
- **MS-13 (#10332):** Dual-hand grip closure constraint protocol.
- **MS-14 (#10333):** Pinocchio `MatchingPlant` full lane (`constraintDynamics` weld, shared contact law, Pink ConstrainedIkReceipt). Contracts + fail-closed blocked receipts landed; native ControlTower run still required for green closure ≤ 1e-4 m.
- **MS-15 (#10334):** Stance detection and contact sphere standoff consistency.
- **MS-20 (#10335):** Pinocchio Pink kinematics trajectory adapter.
- **MS-21 (#10336):** Drake inverse kinematics baseline reference.

---

### Wave 2: Forward Dynamics & Optimal Control (MS-30, MS-31, MS-40 – MS-43, MS-51 – MS-53)

_Objective:_ Contact-aware forward dynamics, ground force equilibrium, and shooting optimization.

- **MS-30 (#10337):** Compliant foot-ground contact model standardization.
- **MS-31 (#10338):** Crocoddyl full-body optimal control action models.
- **MS-40 (#10339):** Computed-torque controller unactuated floating root stabilization.
- **MS-41 (#10340):** ZMP dynamics filter with support polygon constraint.
- **MS-42 (#10341):** OpenSim Moco whole-body tracking problem.
- **MS-43 (#10342):** Drake multibody plant forward simulation integration.
- **MS-51 (#10344):** MyoSuite neural controller fail-closed contract.
- **MS-52 (#10345):** Simscape physical reference replay parity verification.
- **MS-53 (#10346):** Ground reaction force compliance audit.

---

### Wave 3: Cross-Engine Verification & Full-Body Integration (MS-60 – MS-62, MS-70 – MS-72)

_Objective:_ Cross-validation across all six engines with step-size convergence and torque parity.

- **MS-60 (#10347):** Simscape R2025b run management — scripted replay, run manifest, committed run-102 candidate/GIF (owner-authorized contract revision 2026-09-17).
- **MS-61 (#10348):** Qualify Simscape R2025b topology + full-marker terminal disclosure (run-103 scaffolding; native Fit blocked without licensed host / neck model via MS-104).
- **MS-62 (#10349):** Dynamic torque consistency audit.
- **MS-70 (#10350):** Dual-club driver and 7-iron capture suite.
- **MS-71 (#10351):** Full-body mass and inertia matrix parity verification.
- **MS-72 (#10352):** Unactuated floating root consistency check.

---

### Wave 4: GUI, Visualization & Productization (MS-80 – MS-87, MS-90)

_Objective:_ Desktop tools, 3D viewport replay, shadow tracker tile, and results browsing.

- **MS-80 (#10353):** Motion Matching run browser widget in Tools.
- **MS-81 (#10354):** Multi-engine playback synchronization in 3D viewport.
- **MS-82 (#10355):** Interactive GRF and ZMP visualization layer.
- **MS-83 (#10356):** Comparison overlay for swing deviations.
- **MS-85 (#10358):** Export matched swing packages for web replay.
- **MS-86 (#10359):** Benchmark performance profiling and HUD.
- **MS-87 (#10360):** Automated evidence gallery generator.
- **MS-90 (#10361):** End-to-end user smoke journey tests.

---

### Wave 5: Release Qualification & Governance (MS-100 – MS-108)

_Objective:_ Continuous automated qualification and final professional release audit.

- **MS-100 (#10374):** Fail-closed physical acceptance validator.
- **MS-101 (#10375):** Drake native full-body trajectory optimization.
- **MS-102 (#10376):** OpenSim Moco full-body muscle-driven tracking.
- **MS-103 (#10377):** Pinocchio Crocoddyl full-body optimal control integration.
- **MS-104 (#10378):** Driver and 7-iron dual-club G3 coverage across all engines.
- **MS-105 (#10379):** Cross-engine physical convergence and numerical verification.
- **MS-106 (#10380):** Professional release gate and verified matched badge.
- **MS-107 (#10381):** Native automated engine benchmark regression suite.
- **MS-108 (#10382):** Matched swing program end-to-end release audit.

## Status Refresh 2026-09-18

Merged on `main`: MS-01/02/03/04/05/06/10/11/31/50/84/95 (see PRs above); OG-01..09 (#10404, #10414). Open with PRs: MS-21 (#10448, rejected replay), PF-01..04/06/09 (#10442/#10443/#10445/#10447/#10449/#10451, all conflicting and red). Untouched: MS-12..MS-17, MS-20, MS-30, MS-40..43, MS-51..53, MS-60..62, MS-70..72, MS-80..83, MS-85..87, MS-90, MS-100..112. Wave 1 now leads with MS-107 (Pinocchio G1) and MS-100 (fail-closed gates); see the epic #10363 status section of 2026-09-18.
