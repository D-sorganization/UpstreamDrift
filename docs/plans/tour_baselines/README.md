# Tour Baselines Program: Model Identities, Coverage, and Evidence

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584)  
Governing Program: Matched Swing Program ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363))  
Initial Dispatch: [#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585) (TB-00)

---

## 1. Executive Summary and Problem Statement

Prior to this work package, ambiguity existed across several repository subsystems:

1. **Reconstruction Models vs. Dynamic Simulators:** Reconstruction models named `double_pendulum` and `triple_pendulum` (`src/motion_capture/reconstruct/model/registry.py`) track anatomical markers (shoulder-to-wrist and shoulder-to-elbow-to-wrist) but entirely omit the golf club. Their low kinematic residuals in #9914 (33.1 mm driver, 9.6 mm iron) reflect unobserved club endpoints rather than physical swing capture.
2. **Tools Dependency Ownership:** Shipped launcher manifests resolve the `Tools` repository implementation (`double_pendulum_golf`), while local copies and analytical physics engines also exist.
3. **Closed Issue Reconciliation:** Epics #9914, #9921, and #10003 delivered reference overlays, Simscape authority, and OpenSim models respectively, but none qualify full-swing dual-club releases under fail-closed gates G1/G2/G3.

The **Tour Baselines Program** provides a single source of truth across all 13 planned work packages (TB-00 through TB-12), establishing verified model identities, clean alias resolution, and an auditable two-capture coverage matrix across Driver (360 Hz) and 7-Iron (359 Hz) tour datasets.

---

## 2. Program Documentation Index

- [Model Identities and Topologies](model_identities.md) — Comprehensive enumeration of all registered golf models, degree-of-freedom derivation, constraint rank analysis, and club representations.
- [Two-Capture Coverage Matrix](coverage_matrix.md) — Exhaustive model $\times$ {Driver, Iron} matrix, existing artifacts, supported observation sets, and explicit non-golf tool exclusions.
- [Historical Reconciliation and Tools Provenance](reconciliation.md) — Audit of closed issues #9914, #9921, #10003, and submodule pin tracking for `vendor/ud-tools`.
- [Target Audit, Marker Semantics, Events, and Provenance](target_audit.md) — Cryptographic capture contracts, versioned measurement maps, native clocks, inferred event landmarks, and shared subject anatomy.
- [Baseline Packages, Fit Metrics, and Qualification Profiles](qualification_profiles.md) — Versioned baseline package schema, 5 unconflated fit metric statuses, model-specific gates (G1/G2/G3), native replay requirements, and synthetic test guards.

---

## 3. Core Architectural Principles

- **Separation of Model, Backend, Capture, and Evidence:** A model topology (e.g., planar driven double pendulum) is strictly separated from its solver backend (e.g., Scipy ODE vs. MuJoCo Warp), its target capture (Driver vs. 7-Iron), and its qualification status.
- **Fail-Closed Gate Evaluation:** Missing files, missing sensors, or unverified replays fail closed. Partial or reduced-model fits cannot satisfy full-body G3 release obligations.
- **Strict Boundary Ownership:** `Tools` owns `double_pendulum_golf`. UpstreamDrift consumes pinned revisions via `vendor/ud-tools` without in-place monkeypatching.
