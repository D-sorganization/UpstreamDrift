# Publication Gate Checklist for Closed-Loop Null-Space Exploration Results

**Issue:** [UpstreamDrift#12592](https://github.com/D-sorganization/UpstreamDrift/issues/12592) (`NSX-74`)
**Parent Epic:** [UpstreamDrift#12593](https://github.com/D-sorganization/UpstreamDrift/issues/12593) (`NSX: Closed-Loop Null-Space Exploration`)
**Automated Gate Evaluator:** `src/shared/python/loop_nullspace/publication_gate.py` (`evaluate_nsx_publication_gate`)

---

## 1. Purpose and Governance Scope

This document defines the mandatory verification gates required before any result produced by the Closed-Loop Null-Space Exploration (NSX) stack is published externally. It separates **software and mathematical methodology publication readiness** from **empirical physiological claim readiness**.

---

## 2. Mandatory Pre-Publication Checklist

| Gate ID                             | Criterion                                                                                                                                                                                                                                                                                                                     | Verification Artifact                                                                                       | Current Status         |
| :---------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------------------- |
| `GATE-CITATIONS-VERIFIED`           | Every active hypothesis in `src/config/loop_nullspace_registry.json` has `verification_status == "VERIFIED"` with a resolvable archival DOI or URL and section/table reference.                                                                                                                                               | `src/config/loop_nullspace_registry.json`, `docs/research/loop_nullspace/loop_nullspace.tex` (§10, Table 4) | **PASSED**             |
| `GATE-NO-UNVERIFIED-PRIORS`         | Zero `UNVERIFIED` or `REMOVED` citations from the v1 proposal (`CIT-THORNTON-2025`, `CIT-MEHTA-2024`, `CIT-MDPI-GRIP-UNSPECIFIED`, `CIT-CHOI-PARK-2020`, `CIT-SOROKIN-2025`, `CIT-SPAR-UNVERIFIED`, `CIT-BOSCH-DEAD-ZONE`, `CIT-NESBIT-SERRANO-2005`) appear in any active hypothesis statement, UI preset, or default prior. | `src/config/loop_nullspace_registry.json`                                                                   | **PASSED**             |
| `GATE-DV-12554-RECORDED`            | The pre-registered external validation protocol specifying 6-axis dual-section instrumented grips (`hand_on_bare_club`), bilateral force plates, and synchronized surface EMG on held-out golfer swings is committed and tracked.                                                                                             | `docs/development/planning/DV-12554.md`                                                                     | **PASSED**             |
| `GATE-EXPLICIT-NON-CLAIMS`          | All five explicit scientific non-claims (Epic #12593 §8 and `loop_nullspace.tex` §11) are included verbatim in the manuscript and software report metadata.                                                                                                                                                                   | `src/shared/python/loop_nullspace/publication_gate.py` (`EXPLICIT_NON_CLAIMS`)                              | **PASSED**             |
| `GATE-EXTERNAL-PHYSICAL-VALIDATION` | Physical experimental validation on held-out golfer swings under `DV-12554` (net-wrench dynamic balance gate, $95\%$ uncertainty-inflated envelope containment, and NSX-25 sign-certificate concordance).                                                                                                                     | `docs/development/planning/DV-12554.md` (Status: **deferred, needs equipment**)                             | **DEFERRED (`False`)** |

---

## 3. Two-Tier Publication Disposition Rules

1. **Methodology and Software Architecture Publication (`ready_for_methodology_publication == True`):**
   - Permitted when `GATE-CITATIONS-VERIFIED`, `GATE-NO-UNVERIFIED-PRIORS`, `GATE-DV-12554-RECORDED`, and `GATE-EXPLICIT-NON-CLAIMS` all pass.
   - Any published figure or table must label `INVARIANT` quantities as algebraic properties of the rigid-body model and `FAMILY_DEPENDENT` envelopes/allocations as conditional on their declared hypothesis IDs (`H-TORQUE-CAPACITY-BOX`, `H-GRIP-FRICTION-CONE`, `H-EFFORT-CAPACITY-QUADRATIC`, `H-TORQUE-RATE-SMOOTHNESS`).
2. **Empirical Physiological Claim Publication (`ready_for_physiological_claim == False` Until `DV-12554` Closes):**
   - Strictly prohibited until physical instrumentation data from dual 6-axis grip transducers, optical motion capture, dual force plates, and sEMG pass all pre-registered falsification gates in `docs/development/planning/DV-12554.md` on held-out swings.
   - Simulation-generated or inverse-kinematics-derived data may never be used to mark `external_validation_completed = True`.

---

## 4. Explicit Scientific Non-Claims Required in Every External Publication

1. **No Unique Hand Split From Kinematics Alone:** Motion capture and inverse dynamics fix the net bare-club wrench $W_{\mathrm{net}}^{\mathrm{bare}}$ (`0.313 kg` bare club partitioned from `1.1082 kg` welded hands) but never determine a unique 6-D internal hand split $s \in \ker G$ without declared prior hypotheses.
2. **No Stiffness or Co-Contraction Inference From Net Torques:** Net joint torques $\tau$ lie in the quotient modulo the muscle null space $\ker R(q)$ (`NSX-73`) and carry zero information about antagonist co-contraction or joint rotational stiffness.
3. **No Inertial Identification Beyond MOSAIC:** NSX performs zero inertial parameter estimation beyond the identifiable base parameter subspace (`NSX-62`).
4. **No Aerodynamic Resolution of ZTCF Identifiability:** Aerodynamic drag modifies the external club wrench but does not resolve internal loop multiplier redundancy.
5. **No Unverified Literature Priors:** None of the unverified percentage hand-share claims from the v1 proposal are used anywhere in the mathematical specification, software defaults, or user-facing reports.
