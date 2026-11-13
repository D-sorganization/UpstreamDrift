# Physical Acceptance Gates: The Matched Swing Gate Ladder (G1 / G2 / G3)

## Continuation Authority

The single source of truth for numerical thresholds is `AcceptanceGates` and
`evaluate` in `src/shared/python/motion_matching/acceptance.py`. G1 is the
`0` to `0.85 s` dynamic horizon (`g1_whole_rmse_m = 0.025`, `g1_early_rmse_m = 0.012`,
`g1_terminal_rmse_m = 0.035`), not a full-capture IK milestone. Historical reports
must not override `AcceptanceGates`; do not relax thresholds during fitting.
Read the [continuation prompt](AGENT_CONTINUATION_PROMPT.md) and exact receipts.

**Author:** Dieter Olson (`agent:local`)  
**Date:** 2026-09-17 (synchronized with `AcceptanceGates` on 2026-11-12)  
**Issue:** MS-01 (#10322), MS-06 (#10327), Epic #10363, #15015  
**Implementation:** `src/shared/python/motion_matching/acceptance.py`

## MS-61 Full-Marker Terminal Disclosure (#10348)

Simscape (and profiled) receipts that report a terminal metric must disclose
`head_cluster_terminal_rms_m` (`terminal_breakdown` or top-level). The G1
terminal gate always uses the **full-marker** terminal RMS. A
`body_excluding_head_terminal_rms_m` diagnostic may appear under an explicit
profile (`reduced_27_no_neck`) but **cannot** flip full-body acceptance.

## 1. Principles of Acceptance

1. **Physical Reality Over Solver Flags:** Convergence without finite, bounded optical marker and ground-reaction agreement does not pass.
2. **Fail-Closed Evaluation:** Missing channels, unverified hashes, or non-finite values fail closed (`GateStatus.MISSING` / `FAILED`).
3. **No Threshold Relaxation:** Tolerances are fixed across engines; `BOUND_WIDENING > 1.0` or reduced/partial profiles do not satisfy full-body gates.
4. **All Six Engines Required for Release:** G3 release requires MuJoCo, Pinocchio, Drake, OpenSim, Simscape, and MyoSuite.

## 2. Gate Definitions

### Gate G1: Kinematic and Early Horizon Thresholds

| Criterion                 | `AcceptanceGates` Field   | Threshold   | Notes                       |
| ------------------------- | ------------------------- | ----------- | --------------------------- |
| Whole-horizon marker RMSE | `g1_whole_rmse_m`         | 0.025 m     | 25 mm (`0` to `0.85 s`)     |
| Early-window marker RMSE  | `g1_early_rmse_m`         | 0.012 m     | 12 mm initial window        |
| Terminal marker RMSE      | `g1_terminal_rmse_m`      | 0.035 m     | 35 mm full-marker terminal  |
| Club cluster RMSE         | `g1_club_rmse_m`          | 0.060 m     | 60 mm clubhead/shaft        |
| Pelvis yaw RMSE           | `g1_pelvis_yaw_rmse_rad`  | 0.05236 rad | `< 3 deg`                   |
| Pelvis yaw relative error | `g1_pelvis_yaw_error_pct` | 5.0 %       | Relative heading percentage |

### Gate G2: Through-Impact Horizon Thresholds

| Criterion                 | `AcceptanceGates` Field  | Threshold   | Notes                 |
| ------------------------- | ------------------------ | ----------- | --------------------- |
| Whole-horizon marker RMSE | `g2_whole_rmse_m`        | 0.040 m     | 40 mm through impact  |
| Early-window marker RMSE  | `g2_early_rmse_m`        | 0.015 m     | 15 mm initial window  |
| Terminal marker RMSE      | `g2_terminal_rmse_m`     | 0.050 m     | 50 mm terminal window |
| Club cluster RMSE         | `g2_club_rmse_m`         | 0.075 m     | 75 mm club cluster    |
| Pelvis yaw RMSE           | `g2_pelvis_yaw_rmse_rad` | 0.08727 rad | `< 5 deg`             |

### Gate G3: Full-Swing Horizon Thresholds and Release Criteria

| Criterion                | `AcceptanceGates` Field  | Threshold   | Notes                     |
| ------------------------ | ------------------------ | ----------- | ------------------------- |
| Driver whole-swing RMSE  | `g3_whole_driver_rmse_m` | 0.060 m     | 60 mm full driver capture |
| 7-iron whole-swing RMSE  | `g3_whole_iron_rmse_m`   | 0.095 m     | 95 mm full 7-iron capture |
| Early-window marker RMSE | `g3_early_rmse_m`        | 0.020 m     | 20 mm initial window      |
| Terminal marker RMSE     | `g3_terminal_rmse_m`     | 0.080 m     | 80 mm terminal window     |
| Club cluster RMSE        | `g3_club_rmse_m`         | 0.100 m     | 100 mm club cluster       |
| Pelvis yaw RMSE          | `g3_pelvis_yaw_rmse_rad` | 0.10472 rad | `< 6 deg`                 |

| Release Criterion   | Specification                                         | Target                 |
| ------------------- | ----------------------------------------------------- | ---------------------- |
| Engine roster       | MuJoCo, Pinocchio, Drake, OpenSim, Simscape, MyoSuite | 6 / 6 verified         |
| Dual-club coverage  | Tour driver (`driver`) and tour 7-iron (`iron`)       | Both captures verified |
| Cross-engine parity | Joint/marker & generalized force consistency          | `<= 1 mm` / `<= 5%`    |
| Provenance chain    | SHA-256 base spec & receipt integrity                 | Verified hash chain    |

### Shared Physical and Consistency Limits

| Criterion                        | `AcceptanceGates` Field               | Threshold     | Notes                          |
| -------------------------------- | ------------------------------------- | ------------- | ------------------------------ |
| Max normal GRF multiplier        | `max_normal_force_bw_multiplier`      | 3.0 BW        | Peak vertical force limit      |
| Nominal body mass                | `nominal_body_mass_kg`                | 80.0 kg       | Fallback if receipt omits mass |
| Gravitational acceleration       | `gravity_m_s2`                        | 9.80665 m/s^2 | `GRAVITY_M_S2`                 |
| Max ground penetration           | `max_penetration_m`                   | 0.010 m       | 10 mm max sphere intrusion     |
| Max closure translation residual | `max_closure_residual_m`              | 0.005 m       | 5 mm dual-hand weld error      |
| Max closure rotation residual    | `max_closure_residual_rad`            | 0.05 rad      | Dual-hand orientation error    |
| Min ground weight fraction       | `weight_fraction_min`                 | 0.20 BW       | Lower contact loading bound    |
| Max ground weight fraction       | `weight_fraction_max`                 | 3.00 BW       | Upper contact loading bound    |
| Min support polygon fraction     | `min_inside_support_polygon_fraction` | 0.85 fraction | 85% of frames in hull          |
| Max integrator tolerance         | `max_integrator_rtol`                 | 1e-5 rtol     | Declared solver `rtol` bound   |
| Max collocation defect           | `max_collocation_defect_m`            | 0.005 m       | 5 mm dynamic consistency       |
| Max stabilized marker RMSE       | `max_stabilized_marker_rmse_m`        | 0.040 m       | 40 mm low-gain PD tracking     |
| Max friction coefficient         | `max_friction_coefficient`            | 0.80 ratio    | Coulomb friction ratio         |
| Max actuator torque              | `max_actuator_torque_n_m`             | 200.0 N\*m    | Actuator torque bound          |
| Max root residual wrench         | `max_root_residual_n_m`               | 1e-3 N\*m     | Floating-root residual bound   |
| Min club marker coverage         | `min_club_marker_coverage_fraction`   | 0.80 fraction | 80% valid club frames          |

### Horizon Duration Lower Bounds

| Horizon             | `AcceptanceGates` Field | Threshold | Notes                         |
| ------------------- | ----------------------- | --------- | ----------------------------- |
| G1 minimum duration | `g1_min_duration_s`     | 0.85 s    | Top of backswing / transition |
| G2 minimum duration | `g2_min_duration_s`     | 1.20 s    | Through ball impact           |
| G3 minimum duration | `g3_min_duration_s`     | 1.80 s    | Full swing completion         |

## 3. Evaluation Schema and Validation Commands

Programmatic evaluation uses `evaluate` in `src/shared/python/motion_matching/acceptance.py`:

```python
from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    Horizon,
    evaluate,
)

verdict = evaluate(receipt_dict, horizon=Horizon.G1, gates=AcceptanceGates())
```

Run tests:

```bash
pytest tests/unit/motion_matching/test_acceptance.py tests/unit/motion_matching/test_gates_and_coeff_order_15015.py -v
```

## 4. Polynomial Torque Coefficient Order and Pelvis ILC Flag

- **Canonical `A..G` order (ascending powers):** `[A, B, C, D, E, F, G]` evaluates `A + B*t + C*t^2 + D*t^3 + E*t^4 + F*t^5 + G*t^6` (`polynomial_torque.evaluate_polynomial_torque`, `validate_theta.DEFAULT_THETA_BOUND_TABLE`, `CROSS_ENGINE_PARITY_SPEC.md` §2.2).
- **Descending `polyval` order (`constrained_sextic.py`):** Arrays in MATLAB `polyval` convention `[c_6, c_5, c_4, c_3, c_2, c_1, c_0]` (evaluating `c_6*t^6 + ... + c_0`) use `descending_polyval_to_normalized` and `normalized_to_descending_polyval`, reversing columns relative to canonical `A..G`.
- **Pelvis iterative-learning CLI flag:** Use `--pelvis-ilc-passes N` (legacy alias `--shooting-fit N`). Both invoke `dynamics.shooting_fit`, an iterative-learning pelvis-drift update rather than a boundary-value defect-zeroing shooting solver.
