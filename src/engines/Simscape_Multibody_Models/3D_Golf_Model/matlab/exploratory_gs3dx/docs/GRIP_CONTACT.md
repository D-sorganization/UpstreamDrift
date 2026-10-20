# Physical Grip Contact (`GS3DX_GripContact`)

Issue [#13334](https://github.com/D-sorganization/UpstreamDrift/issues/13334) (`HRG-GS-M12`),
epic [#13297](https://github.com/D-sorganization/UpstreamDrift/issues/13297).
Required execution release: **MATLAB R2025b**.
Topology export baseline: [#13333](https://github.com/D-sorganization/UpstreamDrift/issues/13333) (`HRG-GS-M11`, [HUMAN.md](HUMAN.md)).

## What Was Built

In `GS3DX_Human` (`GolfSwing3D_Human.slx`), the club is rigidly welded to the lead hand and closed to the trail hand through a `Weld Joint`, so hand–club and hand–hand contact forces are post-hoc allocations from loop-closure reaction multipliers (`HRG-GS-M11`). `HRG-GS-M12` defines a standalone physical grip-contact fixture in Simscape Multibody (`GS3DX_GripContact`, registered as `gs3dx_names().variants.grip_contact`) that releases the hand–club welds, drives the club through the shared physical pad–grip contact law, and evaluates dynamic net-club-wrench parity against the Pinocchio G6 reference (`simulate_grip_contact` in `src/engines/physics_engines/pinocchio/python/grip_contact_sim.py`).

| Component | `GS3DX_Human` (M11 Welded Loop) | `GS3DX_GripContact` (M12 Physical Contact Fixture) |
| --- | --- | --- |
| Club Joint | Welded to lead hand (`Clubface Vector`) | Free body on `6-DOF Joint` (`Club 6DOF Joint`) with principal inertia frame |
| Trail Hand Closure | `Weld Joint` (`Right Hand Weld`) | Released (**0** `Weld Joint` blocks in model) |
| Hand Motion | Driven by full-body joint tree | Prescribed world poses and twists (`Hand_L`, `Hand_R`) on `Bushing Joint` blocks |
| Contact Law Source | Post-hoc allocation in Python (`simscape_grip_topology.py`) | Codegen-compatible `gs3dx_grip_contact_kernel.m` generated from Python source of truth |
| Contact Law State | Stateless (static allocation per frame) | Stateless algebraic law (no `z_state` input, no `dz_dt` output, no `Integrator` block) |
| Wrench Injection | None (`0 N` injected into welded loop) | World-frame wrench about club origin via `External Force and Torque` (`Club External Wrench`) |

## Canonical Python-to-MATLAB Codegen Kernel (Stateless Law)

To stay 100% DRY with the Python source of truth (`src/shared/python/grip_contact/`), MATLAB never re-authors contact parameters, site coordinates, or constitutive equations by hand:

1. **Generator (`src/shared/python/grip_contact/matlab_kernel.py` and `scripts/export_simscape_grip_kernel.py`):**
   - Extracts all contact law parameters (`normal_stiffness_n_m`, `damping_s_m`, `static_friction`, `dynamic_friction`, `viscous_friction_s_m`, `stribeck_velocity_m_s`, `patch_radius_m`, `spin_transition_rad_s`), club grip frame offsets, and all 3 grip topologies (`overlap`, `interlock`, `ten_finger` at $F_{\mathrm{sq}} = 1100\,\mathrm{N}$ per hand) into fixed-size $(12 \times 17)$ numeric pair tables (`MAX_PHYSICAL_PAIRS = 12`).
   - Emits `tools/generated/gs3dx_grip_contact_kernel.m` (`%#codegen` compliant: fixed sizes, no containers, no strings, no dynamic memory allocation, no `persistent` variables) and `tools/generated/gs3dx_grip_contact_params.json`.
   - `python3 scripts/export_simscape_grip_kernel.py --check` verifies byte-for-byte freshness and fails if the Python contact law, capsule closest-point geometry, or topology parameters drift from the generated `.m` and `.json` artifacts.

2. **Stateless Physical Pair Law (Proven by `pair_contact.py`):**
   - The shared pairwise contact law in `src/shared/python/grip_contact/pair_contact.py`—specifically `sphere_capsule`, `capsule_capsule`, and `_sample_contact` (which delegates to `sphere_ground_contact` and `torsional_friction_moment` in `src/shared/python/motion_matching/contact_law.py`)—is **stateless**: it computes Hunt–Crossley normal force $f_n = \max(0, k_n \delta_+(1 - c_n v_n))$, regularized tanh/Stribeck tangential friction $\vec{f}_t = -\mu(s) f_n \tanh(s / v_{\mathrm{tr}}) \vec{v}_t / s$ with **no tangential spring**, and torsional patch spin moment $\vec{m}_s = -\mu_s f_n r_{\mathrm{patch}} \tanh(\omega_n / \omega_{\mathrm{tr}})\hat{\mathbf{n}}$ purely as algebraic functions of instantaneous penetration $\delta$ and relative velocity $(\vec{v}_{\mathrm{rel}}, \vec{\omega}_{\mathrm{rel}})$.
   - Because the shared law is stateless, `gs3dx_grip_contact_kernel` has no continuous state input (`z_state`) and no state derivative output (`dz_dt`). Its exact signature is:
     ```matlab
     [F_club_w, M_club_w, F_hand_L_w, M_hand_L_w, F_hand_R_w, M_hand_R_w, ...
         pair_normal_n, pair_tangent_n, pair_spin_nm, pair_penetration_m, ...
         pair_slip_speed_m_s, pair_signed_dist_m, pair_force_club_w, ...
         pair_moment_club_w, pair_contact_point_w] = gs3dx_grip_contact_kernel( ...
         p_club, R_club, v_club, omega_club, ...
         p_hand_L, R_hand_L, v_hand_L, omega_hand_L, ...
         p_hand_R, R_hand_R, v_hand_R, omega_hand_R, ...
         style_id)
     ```

3. **Constitutive Golden Vectors (`tests/fixtures/gs3dx_grip_pair_golden.json` and `tests/test_gs3dx_grip_pair_wrench.m`):**
   - Contains 51 constitutive test vectors generated directly from the Python reference across `overlap`, `interlock`, and `ten_finger`, covering 9 named edge cases per style (`separated`, `just_touching`, `near_collinear_capsules`, `stick`, `slip`, `zero_relative_velocity`, `coincident_axis`, `rebounding_clamp`, `zero_rel_vel_comoving`) plus 24 randomized 3D pose/twist states.
   - Verified in Python (`test_golden_vectors_cover_all_styles_and_edge_cases`) and **measured in MATLAB R2025b**: `test_gs3dx_grip_pair_wrench` **PASSED** in R2025b (1 test over all 51 golden cases) to $\le 10^{-10}$ relative and $\le 10^{-9}\,\mathrm{N}$ absolute tolerance.

## Fixture Model Architecture (`GS3DX_GripContact`)

`gs3dx_build_grip_contact` defines `models/GS3DX_GripContact.slx` programmatically from R2025b Simscape Multibody library primitives:

- **World and Gravity:** `Solver Configuration`, `Mechanism Configuration` ($\vec{g} = [0, 0, -9.81]^{\top}\,\mathrm{m/s^2}$, matching the Python spec), and `World Frame`.
- **Free Club (`Club 6DOF Joint`):** `6-DOF Joint` from `Club Init Pose` to `Club_Origin`. Because the club inertia tensor at COM $\mathbf{c}_{\mathrm{club}} = [0.61977, 0, 0]^{\top}\,\mathrm{m}$ has a nonzero off-diagonal product of inertia $I_{xy} = -3.7707\times 10^{-4}\,\mathrm{kg\cdot m^2}$, `scripts/export_simscape_grip_contact_inputs.py` diagonalizes $\mathbf{I}_{\mathrm{com}} = \mathbf{R}_{\mathrm{p}} \operatorname{diag}(I_1, I_2, I_3) \mathbf{R}_{\mathrm{p}}^{\top}$ in Python (`Club COM Transform` + `Club Inertia` with zero `ProductsOfInertia`).
- **Prescribed Hands (`Hand_L`, `Hand_R`):** Each hand is mounted on a `Bushing Joint` (`Hand L Prescribed Joint`, `Hand R Prescribed Joint`) with prescribed motion inputs on `Px, Py, Pz, Rx, Ry, Rz` driven by piecewise Hermite cubic trajectories exported in `gs3dx_grip_contact_scenarios.json`.
- **Sensing, Stateless Kernel, and Wrench Application:** Three `Transform Sensor` blocks (`Club`, `Hand_L`, `Hand_R`) sense world position $p$, rotation matrix $R$, linear velocity $v$, and angular velocity $\omega$ relative to `World`. These feed `Grip Contact Kernel` directly (with no feedback `Integrator` loop) and drive `Club External Wrench` (`External Force and Torque` in `World` axes applied at `Club_Origin`).

## Compiled Block Budget Assessment (`GS3DX_GripContact` vs. `GS3DX_Human`)

The MathWorks Home license enforces a hard limit of **1,000 compiled nonvirtual blocks** (`gs3dx_block_budget(mdl, compiled=true)`), and the GS3DX validation suite reserves **25 blocks** for in-memory diagnostic sensors (`gs3dx_contact_check`), setting a maximum persistent model budget of **975 compiled blocks** (see [BLOCK_BUDGET_FINDINGS.md](BLOCK_BUDGET_FINDINGS.md) and [GROUND_CONTACT.md](GROUND_CONTACT.md)).

### Existing Full-Body Model Budgets (From `BLOCK_BUDGET_FINDINGS.md`)

| Model Variant | Compiled Blocks | Headroom to 1,000 Ceiling | Headroom to 975 Reserve Gate |
| --- | ---: | ---: | ---: |
| `GS3DX_FullBody` | 945 | 55 | 30 |
| `GS3DX_FullBodyContact` | 967 | 33 | 8 |
| `GS3DX_Neck` / `GS3DX_Human` | **975** | **25** | **0** |

### Reasoning-Only Estimate of Block Types Required to Embed Physical Grip Contact in `GS3DX_Human`

> **Estimate (by reasoning only; not compiled or measured in `GS3DX_Human`):** Because `GS3DX_Human` is already at the 975-block persistent reserve ceiling (zero reserve headroom), replacing its welded hand–club loop with physical grip contact would require adding the following block types beyond deleting `Right Hand Weld` (`Weld Joint`):
> - A 6-DOF club joint (`6-DOF Joint`) and principal-inertia alignment frame (`Rigid Transform`) between the lead hand and club body;
> - Three `Transform Sensor` blocks (`Club`, `Hand_L`, `Hand_R` relative to `World`) together with `PS-Simulink Converter` blocks for position, rotation, linear velocity, and angular velocity on each body;
> - The `Grip Contact Kernel` (`MATLAB Function` block) and `GripStyleId` parameter;
> - `External Force and Torque` block(s) on the club (and bilateral hands if reaction wrenches are applied back to the hands) together with `Simulink-PS Converter` and `Demux` routing blocks.
>
> Because `GS3DX_Human` has zero compiled-block headroom below the 975-block reserve gate, physical grip contact is isolated in the standalone `GS3DX_GripContact` fixture model rather than modifying `GS3DX_Human.slx`.

## R2025b Measured Values: Pending Orchestrator Run

Only `test_gs3dx_grip_pair_wrench` has been executed and measured in MATLAB R2025b so far. All `GS3DX_GripContact` model build, compiled block count, simulation, qualification probe, and Pinocchio parity rows are pending the orchestrator's R2025b run of `gs3dx_build_grip_contact` and `gs3dx_grip_contact_run`:

| Check / Metric | Style / Scenario | Gate / Tolerance | R2025b Measured Value | Status |
| --- | --- | --- | --- | --- |
| Golden test result (`test_gs3dx_grip_pair_wrench`) | All 3 styles, 51 golden cases | $\le 10^{-10}$ rel / $\le 10^{-9}\,\mathrm{N}$ abs | **1/1 test PASSED in R2025b (all 51 golden cases)** | **Passed** |
| Compiled block count (`GS3DX_GripContact`) | Uncompiled / compiled total | $\le 975$ compiled ($1{,}000$ ceiling $- 25$ reserve) | Pending orchestrator run | Pending |
| Net-club-wrench parity vs. Pinocchio (`net_force` peak / RMS, `net_moment_com` peak / RMS) | `overlap` / `hold` | Peak $\le 5\%$, RMS $\le 2\%$ | Pending orchestrator run | Pending |
| Net-club-wrench parity vs. Pinocchio (`net_force` peak / RMS, `net_moment_com` peak / RMS) | `overlap` / `short_swing` | Peak $\le 5\%$, RMS $\le 2\%$ | Pending orchestrator run | Pending |
| Net-club-wrench parity vs. Pinocchio (`net_force` peak / RMS, `net_moment_com` peak / RMS) | `interlock` / `hold` | Peak $\le 5\%$, RMS $\le 2\%$ | Pending orchestrator run | Pending |
| Net-club-wrench parity vs. Pinocchio (`net_force` peak / RMS, `net_moment_com` peak / RMS) | `interlock` / `short_swing` | Peak $\le 5\%$, RMS $\le 2\%$ | Pending orchestrator run | Pending |
| Net-club-wrench parity vs. Pinocchio (`net_force` peak / RMS, `net_moment_com` peak / RMS) | `ten_finger` / `hold` | Peak $\le 5\%$, RMS $\le 2\%$ | Pending orchestrator run | Pending |
| Net-club-wrench parity vs. Pinocchio (`net_force` peak / RMS, `net_moment_com` peak / RMS) | `ten_finger` / `short_swing` | Peak $\le 5\%$, RMS $\le 2\%$ | Pending orchestrator run | Pending |
| Timestep-convergence change (`max_rel_force_diff`) | `overlap` / `short_swing` | $\le 2\%$ relative force change | Pending orchestrator run | Pending |
| Save/reload difference (`max_abs_force_diff_n`, `max_abs_moment_diff_nm`) | `overlap` / `short_swing` | $\le 10^{-11}\,\mathrm{N}$ / $\mathrm{N\cdot m}$ | Pending orchestrator run | Pending |
| Frame-probe error at $t = 0$ (`max_pos_err_m`, `max_rot_err_fro`) | All styles & scenarios | $\le 10^{-8}\,\mathrm{m}$, $\le 10^{-8}$ Frobenius | Pending orchestrator run | Pending |
| Force-sign check (`min_normal_push_dot_n`) | All styles (`hold`) | $> 0\,\mathrm{N}$ outward normal push | Pending orchestrator run | Pending |

## Validation and Reproduction Commands

### Python Generator, Freshness, and Unit Verification (Headless Linux / CI)

```bash
python3 scripts/export_simscape_grip_kernel.py --check
python3 scripts/export_simscape_grip_contact_inputs.py --check
python3 -m pytest tests/unit/grip_contact/test_simscape_grip_contact.py
```

### MATLAB R2025b Execution and Receipt Generation (R2025b Host)

```matlab
% 1. Verify constitutive golden vectors (51 cases across 3 styles):
addpath('src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools');
addpath('src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tools/generated');
addpath('src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/tests');
results = runtests('test_gs3dx_grip_pair_wrench');
assertSuccess(results);

% 2. Build GS3DX_GripContact and run all 6 style/scenario simulations + qualification probes:
repo_root = pwd;
out_dir = fullfile(repo_root, ...
    'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/receipts/grip_contact');
summary = gs3dx_grip_contact_run(repo_root, out_dir);
```
