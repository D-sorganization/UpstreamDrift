# Physical Acceptance Gates: The Matched Swing Gate Ladder (G1 / G2 / G3)

## Continuation Review Warning

The 2026-09-18 source review found that this document's `acceptance_contract.py`
reference does not exist on reviewed main. The implemented evaluator is
`src/shared/python/motion_matching/acceptance.py`: G1 is the 0–0.85 s dynamic
horizon (whole 25 mm, early 12 mm, terminal 35 mm), not a 30 mm full-capture IK
milestone. Historical tables below conflict with that evaluator and epic
#10363 and MUST NOT be used to accept a run. MS-100 (#10374) must reconcile
and version the complete contract; do not change thresholds during fitting.
Read the [continuation prompt](AGENT_CONTINUATION_PROMPT.md) and exact receipts.
The review does not qualify the evaluator or replace missing physical evidence.

**Author:** Dieter Olson (`agent:local`)  
**Date:** 2026-09-17  
**Issue:** MS-01 (#10322), MS-06 (#10327), Epic #10363  
**Implementation:** `src/shared/python/motion_matching/acceptance.py`

---

> **Single authority (2026-09-18):** the numerical thresholds in force are the ones coded in `src/shared/python/motion_matching/acceptance.py` (`AcceptanceGates`). Where this document differs, the code wins until MS-100 versions the gates; this document is being aligned.

## MS-61 Full-Marker Terminal Disclosure (#10348)

Simscape (and profiled) receipts that report a terminal metric must also disclose
`head_cluster_terminal_rms_m` (via `terminal_breakdown` or top-level). The G1
terminal gate always uses the **full-marker** terminal RMS. A
`body_excluding_head_terminal_rms_m` diagnostic may appear under an explicit
reduced-model profile (`reduced_27_no_neck`) but **cannot** flip full-body
acceptance. Head-marker exclusion and undocumented threshold relaxation fail
closed. See `full_marker_terminal.py` and run-103 scaffolding under
`docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_103/`.

## 1. Principles of Acceptance

1. **Physical Reality Over Structural Flags:** A run does not pass a gate merely because a solver converged without raising an exception. Physical acceptance requires finite, bounded error against ground-truth optical marker data and ground reaction force laws.
2. **Fail-Closed Evaluation:** Missing evidence, unverified hashes, missing sensor channels, or non-finite values fail closed immediately with an explicit rejection reason.
3. **No Threshold Relaxation:** Numerical tolerances are fixed across all engines and captures. Milestones achieved with widened joint limits (`BOUND_WIDENING > 1.0`), strength scaling, or reduced/partial models do not satisfy full-body release gates.
4. **All Six Engines Required for Release:** Full release (G3) requires all six fleet engines: MuJoCo, Pinocchio, Drake, OpenSim, Simscape, and MyoSuite.

---

## 2. Gate Definitions

### Gate G1: Kinematic Tracking Gate (Engineering Milestone)

Evaluates marker-level tracking precision and physiological realism of the inverse kinematics reference.

| Criterion                            | Metric / Field                    | Required Threshold                     | Rationale                                                     |
| ------------------------------------ | --------------------------------- | -------------------------------------- | ------------------------------------------------------------- |
| **Whole-Capture Marker RMS**         | `ik.marker_rms_m`                 | $\le 0.030\text{ m}$ ($30\text{ mm}$)  | Sub-30 mm full-swing kinematic capture agreement              |
| **Address Pose Marker RMS**          | `address.calibrated.marker_rms_m` | $\le 0.010\text{ m}$ ($10\text{ mm}$)  | Accurate initial stance placement at frame 0                  |
| **Joint Limit Violations**           | `ik.range_of_motion_flags`        | $0\text{ violations}$                  | Strict human anatomical joint limits (`BOUND_WIDENING = 1.0`) |
| **Stance Sphere Ground Penetration** | `ik.lowest_sphere_height_min_m`   | $\ge -0.005\text{ m}$ ($-5\text{ mm}$) | No unphysical ground clipping during swing                    |
| **Closure Translation Residual**     | `ik.closure_error_max_m`          | $\le 0.005\text{ m}$ ($5\text{ mm}$)   | Dual-hand grip closure maintained at address and impact       |

> [!NOTE]
> G1 is an engineering kinematic milestone, not a dynamic match or product release.

---

### Gate G2: Dynamic Ground Support Gate (Floating Root & GRF Compliance)

Evaluates forward dynamic simulation with an unactuated floating pelvis root, compliant foot sole contact, and computed-torque tracking.

| Criterion                          | Metric / Field                                       | Required Threshold                     | Rationale                                                    |
| ---------------------------------- | ---------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| **Whole-Run Dynamic Marker RMS**   | `dynamics.marker_rms_m`                              | $\le 0.120\text{ m}$ ($120\text{ mm}$) | Dynamic tracking error bounded across full swing             |
| **Floating Root Tracking Error**   | `dynamics.root_tracking_rms_m`                       | $\le 0.050\text{ m}$ ($50\text{ mm}$)  | Pelvis position maintained without artificial root actuators |
| **Support Polygon Fraction**       | `dynamics.inside_support_polygon_fraction`           | $\ge 0.90$ ($90\%$)                    | Center of pressure / ZMP remains inside foot base of support |
| **Vertical Ground Reaction Force** | `dynamics.reference_zmp.vertical_grf_over_weight[0]` | $\ge 0.0$ ($F_z \ge 0$)                | Unilateral contact only; feet cannot pull on the ground      |
| **Peak Joint Torques**             | `dynamics.peak_joint_torque_n_m`                     | $\le 2000\text{ N}\cdot\text{m}$       | Physically bounded muscular effort                           |

---

### Gate G3: Multi-Engine Cross-Validation & Professional Release

Evaluates complete cross-engine parity and dual-club coverage across the entire fleet.

| Criterion                         | Specification                                                        | Threshold                                |
| --------------------------------- | -------------------------------------------------------------------- | ---------------------------------------- |
| **Engine Roster**                 | All 6 engines: MuJoCo, Pinocchio, Drake, OpenSim, Simscape, MyoSuite | 6 / 6 verified                           |
| **Dual-Club Coverage**            | Tour Driver (`driver`) and Tour 7-Iron (`iron`)                      | Both captures verified                   |
| **Cross-Engine Kinematic Parity** | Inter-engine joint and marker difference on identical spec           | $\le 0.001\text{ m}$ ($1\text{ mm}$) RMS |
| **Cross-Engine Dynamic Parity**   | Inter-engine generalized force and acceleration difference           | $\le 5\%$ relative error                 |
| **Provenance Chain**              | Verified table and base spec SHA-256 in every receipt                | 100% hash agreement                      |
| **Release Verdict**               | Fail-closed gate review and artifact qualification                   | Signed off by release auditor            |

---

## 3. Evaluation Schema & Validation Commands

Validation is executed programmatically via `src/shared/python/motion_matching/acceptance.py`:

```python
from src.shared.python.motion_matching.acceptance_contract import (
    evaluate_receipt_acceptance,
)

verdict = evaluate_receipt_acceptance(receipt_dict)
if not verdict.passed:
    print(f"Gate {verdict.gate} REJECTED: {verdict.reason}")
```

Run test suite:

```bash
pytest tests/unit/motion_matching/test_acceptance.py -v
```
