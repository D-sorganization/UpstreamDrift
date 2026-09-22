# Ground-Support Pipeline Receipts Reference

This document defines the formal schema, units, meanings, and originating pipeline stages for all fields stored in `receipt.json` produced by the ground-support pipeline (`src/shared/python/motion_matching/pipeline/`).

Downstream tools (launcher tile, feature parity, MJX trajectory optimization) read these fields by name according to this contract.

Regenerate with:

```bash
python -m src.shared.python.motion_matching.pipeline.receipt_schema --markdown
```

## Top-Level Metadata

Summary provenance, hashes, file references, and execution duration written by `pipeline.receipt.build_ground_support_receipt`.

| Field                      | Unit     | Meaning                                                                                                | Stage    |
| :------------------------- | :------- | :----------------------------------------------------------------------------------------------------- | :------- |
| `backend`                  | string   | Kinematic backend engine used for tracking (mujoco or pink)                                            | metadata |
| `engine`                   | string   | Full-body dynamics and plant engine (mujoco, drake, pinocchio)                                         | metadata |
| `base_spec_sha256`         | hash     | SHA256 hash of the initial input model specification document                                          | metadata |
| `base_spec_file`           | filename | Filename of the input model specification document                                                     | metadata |
| `spec_file`                | filename | Filename of the final scaled and calibrated spec document                                              | metadata |
| `hipcal_spec_file`         | filename | Filename of the intermediate hip-calibrated spec document                                              | metadata |
| `recalibrate_upper`        | bool     | Whether upper-body marker placements were recalibrated                                                 | metadata |
| `anthropometric`           | m, kg    | Subject stature (m) and mass (kg) if anthropometric geometry                                           | metadata |
| `de_leva_table_sha256`     | hash     | SHA256 hash of the de Leva anthropometric table if anthropometric geometry                             | metadata |
| `posture_top_of_backswing` | compound | Upper body posture metrics at top of backswing (~0.83 s)                                               | metadata |
| `spec_sha256`              | hash     | Raw-file SHA256 of the final scaled specification bytes (receipt-provenance-chain/1)                   | metadata |
| `spec_canonical_sha256`    | hash     | Canonical JSON SHA256 of the final scaled specification (receipt-provenance-chain/1 semantic identity) | metadata |
| `hip_calibration`          | compound | Functional hip calibration metrics and alignment                                                       | metadata |
| `candidate_sha256`         | hash     | SHA256 hash of candidate artifact byte stream                                                          | metadata |
| `capture`                  | text     | Optical motion capture trial name (driver or iron)                                                     | metadata |
| `capture_sha256`           | hash     | SHA256 hash of raw C3D optical motion capture file                                                     | metadata |
| `club`                     | compound | Club inertial properties, dimensions, and grip offset                                                  | metadata |
| `grip_rotation_deg`        | deg      | Fitted constant grip roll rotations for lead and trail hands                                           | metadata |
| `wrists_bounded`           | bool     | Whether wrist and forearm joints were bounded to human ranges                                          | metadata |
| `labels`                   | names    | List of tracked optical marker labels in kinematic model                                               | metadata |
| `ground`                   | compound | Ground height, toe spheres, and stance detection results                                               | ground   |
| `address`                  | compound | Address pose fit, static trial, and initial stance posture                                             | address  |
| `ik`                       | compound | Full-trajectory kinematic matching and marker calibration                                              | ik       |
| `dynamics`                 | compound | Forward dynamics tracking, computed torque, and ZMP diagnostics                                        | dynamics |
| `elapsed_s`                | s        | Total wall-clock runtime of ground-support execution                                                   | metadata |
| `qualification`            | text     | Qualification note and status claim for the run                                                        | metadata |
| `acceptance`               | compound | Physical and kinematic acceptance evaluation verdict (MS-01)                                           | metadata |

## Ground Stage (`ground`)

Ground contact plane calibration, toe contact sphere placements, and per-sphere stance detection fractions written by `pipeline.lane.Lane`.

| Field                 | Unit  | Meaning                                                     | Stage  |
| :-------------------- | :---- | :---------------------------------------------------------- | :----- |
| `height_m`            | m     | World z-coordinate of the ground contact plane              | ground |
| `lowest_toe_marker_m` | m     | Lowest toe marker z-coordinate observed across capture      | ground |
| `standoff_m`          | m     | Standoff clearance distance between marker and ground       | ground |
| `policy`              | text  | Policy rule used to calibrate ground height                 | ground |
| `stance_tolerance_m`  | m     | Vertical elevation threshold to consider a sphere in stance | ground |
| `stance_rule`         | text  | Operational definition of ground stance detection           | ground |
| `toe_spheres`         | map   | Definitions of the calibrated toe contact spheres           | ground |
| `stance_fraction`     | ratio | Fraction of trajectory frames each sphere spends in contact | ground |

## Address Stage (`address`)

Address pose calibration, static neutral trial placement, CoM support check, and optional closure weld optimization written by `pipeline.address`.

| Field            | Unit     | Meaning                                                     | Stage   |
| :--------------- | :------- | :---------------------------------------------------------- | :------ |
| `seed_offsets`   | compound | Marker residuals from uncalibrated anatomical seeds         | address |
| `stance_spheres` | names    | Foot contact sphere identifiers in active stance at frame 0 | address |
| `static_trial`   | compound | Static trial neutral spine posture and marker placement     | address |
| `calibrated`     | compound | Final calibrated address pose metrics                       | address |
| `closure_fit`    | compound | Optional closure weld optimization report                   | address |

## Inverse Kinematics Stage (`ik`)

Full-trajectory marker matching, alternating calibration, limb scaling, and range-of-motion flags written by `pipeline.reference`.

| Field                        | Unit       | Meaning                                                      | Stage |
| :--------------------------- | :--------- | :----------------------------------------------------------- | :---- |
| `frames`                     | count      | Total number of trajectory frames matched                    | ik    |
| `marker_rms_m`               | m          | Whole-trajectory marker tracking RMS error                   | ik    |
| `segment_rms_m`              | m          | Per-segment marker RMS error over full trajectory            | ik    |
| `closure_error_max_m`        | m          | Maximum dual-grip weld loop closure residual in IK           | ik    |
| `lowest_sphere_height_min_m` | m          | Minimum foot sphere elevation above ground in raw IK         | ik    |
| `lowest_sphere_height_max_m` | m          | Maximum foot sphere elevation above ground in raw IK         | ik    |
| `attachments_m`              | compound   | Complete mapping of calibrated marker offsets in body frames | ik    |
| `reference`                  | compound   | Properties of smoothed, consistency re-solved reference      | ik    |
| `calibration`                | compound   | Alternating calibration convergence and offsets              | ik    |
| `segment_scaling`            | compound   | Segment length scaling search and optimal parameters         | ik    |
| `joint_ranges_deg`           | deg        | Anatomical lower-limb coordinate bounds in degrees           | ik    |
| `range_of_motion_flags`      | compound   | Excursions exceeding anatomical limits across reference      | ik    |
| `bound_widening`             | multiplier | Safety factor applied to widen joint range limits            | ik    |
| `leg_angle_ranges_deg`       | deg        | Min and max angles observed per lower limb joint coordinate  | ik    |
| `constrained_ik`             | compound   | Optional constrained IK execution diagnostics and provenance | ik    |

## Dynamics Stage (`dynamics`)

Computed-torque tracking simulation, zero-moment point diagnostics, contact parameters, and optional optimization filters written by `pipeline.dynamics`.

| Field                             | Unit     | Meaning                                                          | Stage    |
| :-------------------------------- | :------- | :--------------------------------------------------------------- | :------- |
| `duration_s`                      | s        | Total duration of simulated trajectory                           | dynamics |
| `dt_s`                            | s        | Integration numerical time step                                  | dynamics |
| `controller`                      | compound | Computed torque controller settings                              | dynamics |
| `contact_parameters`              | compound | Ground contact stiffness, dissipation, and friction coefficients | dynamics |
| `reference_zmp`                   | compound | Zero-moment point metrics demanded by reference trajectory       | dynamics |
| `marker_rms_m`                    | m        | Simulated full-body motion marker tracking RMS error             | dynamics |
| `segment_rms_m`                   | m        | Per-segment simulated marker tracking RMS error                  | dynamics |
| `joint_tracking_rms_rad`          | rad      | RMS joint coordinate tracking error between sim and reference    | dynamics |
| `root_tracking_rms_m`             | m        | RMS floating base position tracking error between sim and ref    | dynamics |
| `weight_fraction`                 | BW       | Normalized vertical ground support reaction stats                | dynamics |
| `inside_support_polygon_fraction` | ratio    | Fraction of simulated frames where CoM remains inside support    | dynamics |
| `range_of_motion_flags`           | compound | Range-of-motion excursions observed during simulation            | dynamics |
| `root_error_timeline_m`           | m        | Floating pelvis tracking position error sampled across timeline  | dynamics |
| `backswing_to_1s`                 | compound | Dynamic tracking metrics restricted to backswing (0 to 1.0 s)    | dynamics |
| `peak_joint_torque_n_m`           | N m      | Maximum absolute joint actuator torque exerted in simulation     | dynamics |
| `lowest_sphere_height_min_m`      | m        | Minimum elevation of lowest foot contact sphere in simulation    | dynamics |
| `lowest_sphere_height_max_m`      | m        | Maximum elevation of lowest foot contact sphere in simulation    | dynamics |
| `zmp_filter`                      | compound | Optional cart-table zero-moment-point filter report              | dynamics |
| `shooting_fit`                    | compound | Optional contact-aware shooting fit report                       | dynamics |
| `mjx`                             | compound | Optional MJX differentiable trajectory optimization report       | dynamics |
