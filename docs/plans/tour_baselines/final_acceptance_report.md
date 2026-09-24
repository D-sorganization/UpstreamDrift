# Tour Baselines Final Acceptance Report: Software Integration and Scientific Status

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) (TB-12, [#10597](https://github.com/D-sorganization/UpstreamDrift/issues/10597))  
Governing Program: Matched Swing Program ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363))  
Date: 2026-09-24  
Status: Accepted (Software Architecture Complete; Physical Qualification Tracked Under Governing Epics)

---

## 1. Executive Summary

This report documents the completion of **Epic #10584 ("Tour Baselines Architecture, Model Identities, and Acceptance")**, encompassing child work packages TB-00 through TB-12.

### Critical Acceptance Distinction:

- **Software Integration: 100% Complete & Verified.**  
  All foundational schemas, model registries, coverage matrices, measurement maps, physical error formulas, bounded fit campaigns, qualification evaluators, portable package formats, presenter logic, and GUI widgets are fully implemented, tested, and passing CI gates.
- **Physical Qualification: Explicitly Tracked per Governing Issues.**  
  In strict adherence to the Matched Swing Program mandate, no synthetic success or threshold relaxation was permitted. Mandatory reduced-model pairs (`driven_double_pendulum`, `driven_triple_pendulum`) are qualified and promoted. Full-body models remain accurately labeled with their current scientific status (`G1_KINEMATIC_PASSED`, `NATIVE_CANDIDATE`, `REJECTED`, or `UNAVAILABLE`) and linked directly to their governing epics ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363), [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378), [#10430](https://github.com/D-sorganization/UpstreamDrift/issues/10430), [#10440](https://github.com/D-sorganization/UpstreamDrift/issues/10440)).

---

## 2. Two-Capture Coverage and Qualification Matrix

The authoritative coverage matrix records the state of every registered model across both Driver (360 Hz) and 7-Iron (359 Hz) captures:

| Model ID                        | Capture | Supported | Ownership     | Evidence Status       | Existing Artifact / Blocker                          | Governing Issue                                                         |
| ------------------------------- | ------- | --------- | ------------- | --------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------- |
| `driven_double_pendulum`        | Driver  | Yes       | Tools         | `QUALIFIED`           | Qualified baseline package (RMSE: 14.8 mm)           | [#10589](https://github.com/D-sorganization/UpstreamDrift/issues/10589) |
| `driven_double_pendulum`        | 7-Iron  | Yes       | Tools         | `QUALIFIED`           | Qualified baseline package (RMSE: 16.2 mm)           | [#10589](https://github.com/D-sorganization/UpstreamDrift/issues/10589) |
| `driven_triple_pendulum`        | Driver  | Yes       | Tools         | `QUALIFIED`           | Qualified baseline package (RMSE: 11.4 mm)           | [#10590](https://github.com/D-sorganization/UpstreamDrift/issues/10590) |
| `driven_triple_pendulum`        | 7-Iron  | Yes       | Tools         | `QUALIFIED`           | Qualified baseline package (RMSE: 12.9 mm)           | [#10590](https://github.com/D-sorganization/UpstreamDrift/issues/10590) |
| `constrained_upper_body_golfer` | Driver  | Yes       | Tools         | `REJECTED`            | Planarity failure: > 45 mm out-of-plane distortion   | [#10591](https://github.com/D-sorganization/UpstreamDrift/issues/10591) |
| `constrained_upper_body_golfer` | 7-Iron  | Yes       | Tools         | `REJECTED`            | Planarity failure: > 52 mm out-of-plane distortion   | [#10591](https://github.com/D-sorganization/UpstreamDrift/issues/10591) |
| `full_body_mujoco`              | Driver  | Yes       | UpstreamDrift | `G1_KINEMATIC_PASSED` | `anthro_driver_shoot_g025/receipt.json` (27.3 mm IK) | [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363) |
| `full_body_mujoco`              | 7-Iron  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | `anthro_iron_zmp/receipt.json` (28.6 mm IK)          | [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363) |
| `full_body_simscape`            | Driver  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | Terminal error 40.3 mm > 35 mm gate (MATLAB R2025b)  | [#10440](https://github.com/D-sorganization/UpstreamDrift/issues/10440) |
| `full_body_simscape`            | 7-Iron  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | Terminal error 44.1 mm > 35 mm gate (MATLAB R2025b)  | [#10440](https://github.com/D-sorganization/UpstreamDrift/issues/10440) |
| `full_body_pinocchio`           | Driver  | Yes       | UpstreamDrift | `REJECTED`            | Divergence in dynamic shooting optimizer             | [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) |
| `full_body_pinocchio`           | 7-Iron  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | Pending dual-club G3 cross-engine qualification      | [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) |
| `full_body_drake`               | Driver  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | Pending contact implicit trajectory optimization     | [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) |
| `full_body_drake`               | 7-Iron  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | Pending dual-club G3 cross-engine qualification      | [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) |
| `full_body_opensim`             | Driver  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | Moco full-swing tracking pending adapter             | [#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003) |
| `full_body_opensim`             | 7-Iron  | Yes       | UpstreamDrift | `NATIVE_CANDIDATE`    | Pending dual-club G3 cross-engine qualification      | [#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003) |
| `full_body_myosuite`            | Driver  | No        | UpstreamDrift | `UNAVAILABLE`         | Labeled placeholder; requires MyoSuite environment   | [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) |
| `full_body_myosuite`            | 7-Iron  | No        | UpstreamDrift | `UNAVAILABLE`         | Labeled placeholder; requires MyoSuite environment   | [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) |

---

## 3. End-to-End Acceptance Test Outcomes

The end-to-end acceptance suite (`tests/acceptance/test_tour_baselines_journey.py`) verifies the complete operator and user journey:

1. **Roster Completeness (`test_roster_completeness_across_both_captures`):**  
   All registered models are present, categorized, and accurately badged across both Driver and 7-Iron captures.
2. **Model Detail & Provenance Integrity (`test_model_detail_and_provenance_integrity`):**  
   Every detail view provides compute budgets, parameter dimensions, observed vs fitted marker sets, and governing issues.
3. **Visual Distinction Semantics (`test_visual_distinction_semantics`):**  
   Ensures measured optical markers (`marker_points`) and simulated physics graphics (`continuous_mesh`) maintain strict visual separation.
4. **Session Cloning Isolation (`test_session_cloning_isolation`):**  
   Cloning a baseline into an experimental session folder leaves the canonical baseline package immutable.
5. **Model Comparison Across Complexities (`test_model_comparison_across_complexities`):**  
   Verifies delta computation (RMSE delta, metric shifts, and topological ownership) between models.
6. **Evidence Inspection & Audit Receipts (`test_evidence_inspection_and_audit_receipt`):**  
   Verifies that evidence inspection surfaces git commit SHAs, engine versions, and status bundles.
7. **CLI Reproduction Commands (`test_cli_reproduction_commands`):**  
   Asserts that every model has a copyable, valid reproduction command.
8. **Full UI Widget Journey (`test_full_ui_journey`):**  
   Verifies that the `MotionMatchingWidget` embeds `TourBaselinesWidget`, navigates captures, filters models, and displays detail metadata without exceptions.

**Result: 8 / 8 Acceptance Tests PASSED (100% Green).**

---

## 4. Program Milestones Delivered Across Epic #10584

| Child Package | Issue                                                                   | Deliverable                                                            | Status       |
| ------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------ |
| **TB-00**     | [#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585) | Model identities, alias resolution, 10-model coverage matrix           | CLOSED       |
| **TB-01**     | [#10586](https://github.com/D-sorganization/UpstreamDrift/issues/10586) | Measurement map, native clocks, swing events, canonical targets        | CLOSED       |
| **TB-02**     | [#10587](https://github.com/D-sorganization/UpstreamDrift/issues/10587) | BaselinePackage contract, 5-status bundle, physical 3D error formulas  | CLOSED       |
| **TB-03**     | [#10588](https://github.com/D-sorganization/UpstreamDrift/issues/10588) | Fixed geometry, inertia, and calibrated initial states ($q_0, v_0$)    | CLOSED       |
| **TB-04**     | [#10589](https://github.com/D-sorganization/UpstreamDrift/issues/10589) | Driven double pendulum fit campaign, Driver & 7-Iron qualification     | CLOSED       |
| **TB-05**     | [#10590](https://github.com/D-sorganization/UpstreamDrift/issues/10590) | Driven triple pendulum fit campaign, pronation DoF, dual qualification | CLOSED       |
| **TB-06**     | [#10591](https://github.com/D-sorganization/UpstreamDrift/issues/10591) | Upper body golfer evaluation; documented planarity rejection receipt   | CLOSED       |
| **TB-07**     | [#10592](https://github.com/D-sorganization/UpstreamDrift/issues/10592) | Full-body model audit, receipts, engine versions, governing issues     | CLOSED       |
| **TB-08**     | [#10593](https://github.com/D-sorganization/UpstreamDrift/issues/10593) | Deterministic bounded fit campaigns, parameter bounds, budgets         | CLOSED       |
| **TB-09**     | [#10594](https://github.com/D-sorganization/UpstreamDrift/issues/10594) | Qualification evaluator, adequacy metrics, profile gates               | CLOSED       |
| **TB-10**     | [#10595](https://github.com/D-sorganization/UpstreamDrift/issues/10595) | Discovery service, package import/export, safe model presets           | CLOSED       |
| **TB-11**     | [#10596](https://github.com/D-sorganization/UpstreamDrift/issues/10596) | Motion matching presenter, UI widget, session actions, replay          | CLOSED       |
| **TB-12**     | [#10597](https://github.com/D-sorganization/UpstreamDrift/issues/10597) | User guide, operator runbook, acceptance suite, final report           | **ACCEPTED** |

---

## 5. Architectural Health and Verification

All code and documentation additions meet the repository's strict architecture standards:

- **File Size Budget:** All files strictly under 1,200 lines (`check_file_size_budget.py`).
- **Function Line Budget:** All functions strictly under 100 lines (`check_architecture_budget.py`).
- **LOD (Law of Demeter):** Zero deep traversals in presenter or widget modules (`check_lod.py`).
- **Ruff & Formatting:** 100% clean across `src/`, `tests/`, and `docs/`.
- **Typing:** Strict mypy compliance with zero missing annotations.

---

## 6. Handoff and Next Steps

With the completion of TB-12, **Epic #10584 is ready for closure**. Ongoing scientific fitting for full-body models continues under their respective governing issues:

- **MuJoCo Ground Support:** [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363)
- **Multi-Engine Trajectory Optimization (Pinocchio / Drake):** [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378)
- **Simscape Tour Matching Authority (MATLAB R2025b):** [#10440](https://github.com/D-sorganization/UpstreamDrift/issues/10440)
- **OpenSim Moco Musculoskeletal Fit:** [#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003)
