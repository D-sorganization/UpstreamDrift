# Historical Reconciliation and Tools Repository Provenance

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584)  
Governing Issue: [#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585) (TB-00)

---

## 1. Reconciliation of Closed Issues

A critical requirement of the Matched Swing Program is that issue closure in GitHub does not automatically signify scientific or physical full-swing qualification under fail-closed gates G1/G2/G3.

### 1.1 Issue #9914: C3D Reference Fitting Epic (Closed)

- **Governing Issue:** [#9914](https://github.com/D-sorganization/UpstreamDrift/issues/9914)
- **Shipped Branch / PR:** `feat/c3d-reference-overlay-9914` / PR #9918 (`6f2d63325f`)
- **Reported Numbers:**
  - `double_pendulum`: 33.10 mm driver RMS / 26.86 mm iron RMS
  - `triple_pendulum`: 9.60 mm driver RMS / 8.01 mm iron RMS
  - `golfer`: 48.51 mm driver RMS / 49.69 mm iron RMS
- **Reconciliation Finding:**
  The `double_pendulum` and `triple_pendulum` in #9914 are purely kinematic reconstruct models that omit the golf club entirely and track only shoulder-to-wrist and shoulder-to-elbow-to-wrist marker positions. Their seemingly low error is an artifact of having unconstrained, unobserved distal extremities without club inertial dynamics. #9914 successfully delivered reference visualization overlays, but does not provide forward-dynamics or torque-driven baseline models.

### 1.2 Issue #9921: Simscape Tour Matching (Closed)

- **Governing Issue:** [#9921](https://github.com/D-sorganization/UpstreamDrift/issues/9921)
- **Scope:** Simscape 3D Golf Model matching in MATLAB R2025b.
- **Reported Numbers:**
  - Run-102 (0 to 0.85 s window): 20.3 mm whole-swing RMS.
  - Terminal impact error: 40.3 mm (exceeded the 35.0 mm terminal gate).
- **Reconciliation Finding:**
  Simscape serves as the historical tour capture authority and cross-validation reference, strictly pinned to MATLAB R2025b. Because terminal error exceeded the gate, it remains an authoritative baseline lane but not a released G3 full-swing match.

### 1.3 Issue #10003: OpenSim Tour Matching (Closed)

- **Governing Issue:** [#10003](https://github.com/D-sorganization/UpstreamDrift/issues/10003)
- **Scope:** OpenSim Moco musculoskeletal tracking.
- **Reported Numbers:**
  - Calibration rungs (0.10 s, 0.30 s): 41 mm - 42 mm.
  - 0.60 s full solve: Converged, but open-loop replay diverged to 81 mm whole-swing RMS and 204 mm terminal error.
- **Reconciliation Finding:**
  Model architecture and coordinate definitions merged under #10414 (OG-01..09). Open-loop replay divergence prevents G1 promotion; full-swing tracking remains active under MS-102 (#10376).

---

## 2. Tools Submodule Revision Tracking and Ownership Boundary

The review identified a previous discrepancy between the gitlink pointer and local checkouts:

- **Historical Review Gitlink:** `62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1`
- **Current Verified HEAD Submodule Commit:** `a9ed0e7c5c6905b1164082659051d6381068052d`
- **Submodule Path:** `vendor/ud-tools`
- **Remote Source:** `https://github.com/D-sorganization/Tools.git`

### Ownership and Contribution Rules:

1. **Tools Owns `double_pendulum_golf`:** The 2-DOF and 3-DOF pendulum simulators and swing objective tools belong to the `Tools` repository.
2. **No In-Place Modifications:** Contributors must never directly edit files inside `vendor/ud-tools`.
3. **Update Protocol:** Any bug fixes or algorithmic enhancements must be made as a pull request in the `Tools` repository. Once merged, `UpstreamDrift` updates its submodule gitlink via a reviewed pull request.

---

## 3. Governing Full-Body Epics

Work on reduced models under #10584 must not duplicate or bypass the governing full-body issues:

| Issue                                                                   | Title                                                  | Accountable Role             | Mandate                                                                                             |
| ----------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------- | --------------------------------------------------------------------------------------------------- |
| [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363) | Matched Swing Program                                  | Program Lead (`agent:local`) | Single source of truth for physical gates G1, G2, G3 across all 6 physics engines.                  |
| [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378) | MS-104: Driver and 7-Iron Dual-Club G3 Coverage        | Full-Body Lead               | Mandatory dual-club qualification across MuJoCo, Pinocchio, Drake, OpenSim, Simscape, and MyoSuite. |
| [#10430](https://github.com/D-sorganization/UpstreamDrift/issues/10430) | Multi-Engine Trajectory Service & Replay Qualification | Verification Lead            | Independent forward replay verification for all candidate control trajectories.                     |
