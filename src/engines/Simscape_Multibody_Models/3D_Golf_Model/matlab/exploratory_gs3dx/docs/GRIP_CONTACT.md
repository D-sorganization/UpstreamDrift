# Physical Grip Contact (`GS3DX_GripContact`)

Issue [#13334](https://github.com/D-sorganization/UpstreamDrift/issues/13334) (`HRG-GS-M12`),
epic [#13297](https://github.com/D-sorganization/UpstreamDrift/issues/13297).
Required execution release: **MATLAB R2025b**.
Topology export baseline: [#13333](https://github.com/D-sorganization/UpstreamDrift/issues/13333) (`HRG-GS-M11`, [HUMAN.md](HUMAN.md)).

## What Was Built

In `GS3DX_Human` (`GolfSwing3D_Human.slx`), the club is rigidly welded to the lead hand and closed to the trail hand through a `Weld Joint`, so hand–club and hand–hand contact forces are post-hoc allocations from loop-closure reaction multipliers (`HRG-GS-M11`). `HRG-GS-M12` builds a physical grip-contact mode in Simscape Multibody (`GS3DX_GripContact`, registered as `gs3dx_names().variants.grip_contact`) that releases the hand–club welds, drives the club solely through the shared physical contact law and bilateral hand–club bushings, and evaluates dynamic net-club-wrench parity against the Pinocchio G6 reference (`ClubInHands` in `src/engines/physics_engines/pinocchio/python/grip_contact_sim.py`).

| Component | `GS3DX_Human` (M11 Welded Loop) | `GS3DX_GripContact` (M12 Physical Contact Fixture) |
| --- | --- | --- |
| Club Joint | Welded to lead hand (`Clubface Vector`) | Free body on `6-DOF Joint` (`Club 6DOF`) with principal inertia frame |
| Trail Hand Closure | `Weld Joint` (`Right Hand Weld`) | Released (**0** `Weld Joint` blocks in model) |
| Hand Motion | Driven by full-body joint tree | Prescribed world poses and twists (`Hand_L`, `Hand_R`) on `Bushing Joint` blocks |
| Contact Law Source | Post-hoc allocation in Python (`simscape_grip_topology.py`) | Codegen-compatible `gs3dx_grip_contact_kernel.m` generated from Python source of truth |
| State Integration | None (static allocation per frame) | Explicit `dz_dt` port integrated by Simulink `Integrator` block (`Slip State Integrator`) |
| Wrench Injection | None (`0 N` injected into welded loop) | World-frame wrench about club origin via `External Force and Torque` (`Club External Wrench`) |

## Canonical Python-to-MATLAB Codegen Kernel

To stay 100% DRY with the Python source of truth (`src/shared/python/grip_contact/` and `src/shared/python/motion_matching/contact_kernel.py`), MATLAB never re-authors contact parameters, site coordinates, or constitutive equations by hand:

1. **Generator (`src/shared/python/grip_contact/matlab_kernel.py` and `scripts/export_simscape_grip_kernel.py`):**
   - Extracts all contact parameters (`HuntCrossleyParams`, `FrictionParams`, `SpinFrictionParams`, `BushingParameters`, `GripGeometry`) and all 3 grip topologies (`overlap`, `interlock`, `ten_finger`) into fixed-size numeric tables (`N_SITES = 16`, `N_HG_MAX = 10`, `N_HH_MAX = 2`).
   - Emits `tools/generated/gs3dx_grip_contact_kernel.m` (`%#codegen` compliant: fixed sizes, no containers, no strings, no dynamic memory allocation, no `persistent` variables) and `tools/generated/gs3dx_grip_contact_params.json`.
   - `python3 scripts/export_simscape_grip_kernel.py --check` verifies byte-for-byte freshness and fails if the Python contact law, capsule closest-point geometry (including any update from PR #13457), or topology parameters drift from the generated `.m` and `.json` artifacts.

2. **Physical Pair Law and Explicit State Derivative (`dz_dt`):**
   - Evaluates sphere–cylinder (`palm_heel`, `thenar`, `thumb_pad`, `index_pad`, `middle_pad`, `ring_pad`) and capsule–cylinder (`little_finger` on `lead` and `trail`) closest points on the club grip cylinder ($r_{\mathrm{grip}} = 0.014\,\mathrm{m}$, $L_{\mathrm{grip}} = 0.27\,\mathrm{m}$), plus sphere–sphere / sphere–capsule / capsule–capsule closest points for diagnostic hand–hand pairs.
   - Evaluates the qualified Hunt–Crossley normal law $f_n = \max(0, k\delta_+(1 - c v_n))$, regularized Coulomb–Stribeck tangential friction $\vec{f}_t = -\mu(s) f_n \tanh(s / v_{\mathrm{tr}}) \vec{v}_t / s_{\mathrm{safe}}$, and torsional patch spin moment $m_s = -\mu_s f_n r_{\mathrm{patch}} \tanh(\omega_n / \omega_{\mathrm{tr}})$.
   - Evaluates the bilateral translational and rotational grip bushings (`default_bushing()`: $k_t = 8000\,\mathrm{N/m}$, $c_t = 160\,\mathrm{N\cdot s/m}$, $k_r = (80, 150, 150)\,\mathrm{N\cdot m/rad}$, $c_r = (1.6, 3.0, 3.0)\,\mathrm{N\cdot m\cdot s/rad}$) at the calibrated left and right grip anchor frames ($x_{\mathrm{lead}} = -0.0725\,\mathrm{m}$, $x_{\mathrm{trail}} = 0.0\,\mathrm{m}$).
   - Exposes the explicit continuous-time tangential slip-speed state derivative $\dot{z}_k = \|\vec{v}_{t,k}\|_2$ (`dz_dt`, $10\times 1$) so Simulink integrates cumulative slip displacement $z_k(t) = \int_0^t \|\vec{v}_{t,k}(\tau)\|_2\,d\tau$ through a standard `Integrator` block without `persistent` state.

3. **Constitutive Golden Vectors (`tests/fixtures/gs3dx_grip_pair_golden.json` and `tests/test_gs3dx_grip_pair_wrench.m`):**
   - Contains 51 constitutive test vectors generated directly from the Python reference kernels across `overlap`, `interlock`, and `ten_finger`, covering 9 named edge cases per style (`separated`, `just_touching`, `near_collinear_capsules`, `stick`, `slip`, `zero_relative_velocity`, `coincident_axis`, `rebounding_clamp`, `calibrated_hold_t0`) plus 24 randomized 3D pose/twist states.
   - Verified in Python (`test_constitutive_golden_vectors_match_matlab_kernel_reference`) to $0.0\,\mathrm{N}$ / $0.0\,\mathrm{N\cdot m}$ difference and in MATLAB R2025b (`test_gs3dx_grip_pair_wrench.m`) to $\le 10^{-10}$ relative and $\le 10^{-9}\,\mathrm{N}$ absolute tolerance.

## Fixture Model Architecture (`GS3DX_GripContact`)

`gs3dx_build_grip_contact` builds `models/GS3DX_GripContact.slx` programmatically from R2025b Simscape Multibody library primitives:

- **World and Gravity:** `Solver Configuration`, `Mechanism Configuration` ($\vec{g} = [0, 0, -9.81]^{\top}\,\mathrm{m/s^2}$, matching Pinocchio `model.gravity.linear`), and `World Frame`.
- **Free Club (`Club 6DOF`):** `6-DOF Joint` (spherical quaternion + 3D Cartesian translation) from `World` to `Club_Origin`. Because the Pinocchio club inertia tensor at COM $\mathbf{c}_{\mathrm{club}} = [0.61977, 0, 0]^{\top}\,\mathrm{m}$ has a nonzero off-diagonal product of inertia $I_{xy} = -3.7707\times 10^{-4}\,\mathrm{kg\cdot m^2}$ while R2025b `Inertia` blocks with `InertiaType = 'Custom'` require Moments + Products of Inertia to stay under the Home license block budget without adding extra port blocks, `gs3dx_export_scenario_inputs` diagonalizes $\mathbf{I}_{\mathrm{com}} = \mathbf{R}_{\mathrm{p}} \operatorname{diag}(I_1, I_2, I_3) \mathbf{R}_{\mathrm{p}}^{\top}$ in Python (`Club_Principal_Transform` + `Club_Inertia` with zero `ProductsOfInertia`).
- **Prescribed Hands (`Hand_L`, `Hand_R`):** Each hand is mounted on a `Bushing Joint` (`Hand_L_Motion`, `Hand_R_Motion`) with `MotionActuationMode = 'ProvidedByInput'` on `Px, Py, Pz, Rx, Ry, Rz`. `scripts/export_simscape_grip_contact_inputs.py` exports exact 12-interval piecewise Hermite cubic coefficients (`C3, C2, C1, C0`) on $[0, 0.10]\,\mathrm{s}$ so the `Prescribed Hand Trajectory` MATLAB Function block supplies analytic $[q(t), \dot{q}(t), \ddot{q}(t)]$ with $0.0\,\mathrm{m}$ position error and $< 5.1\times 10^{-7}\,\mathrm{m/s}$ velocity error across the 51-sample $2\,\mathrm{ms}$ grid.
- **Sensing, Kernel, and Wrench Application:** Three `Transform Sensor` blocks (`Sensor_Club`, `Sensor_Hand_L`, `Sensor_Hand_R`) sense world position $p$, rotation matrix $R$, linear velocity $v$, and angular velocity $\omega$ relative to `World`. These feed `Grip Contact Kernel`, which integrates `dz_dt` through `Slip State Integrator` and drives `Club External Wrench` (`External Force and Torque` with `ForceResolutionFrame = 'World'` and `TorqueResolutionFrame = 'World'` applied at `Club_Origin`).

## Compiled Block Budget Assessment (`GS3DX_GripContact` vs. `GS3DX_Human`)

The MathWorks Home license enforces a hard limit of **1,000 compiled nonvirtual blocks** (`gs3dx_block_budget(mdl, compiled=true)`), and the GS3DX validation suite reserves **25 blocks** for in-memory diagnostic sensors (`gs3dx_contact_check`), setting a maximum persistent model budget of **975 compiled blocks** (see [BLOCK_BUDGET_FINDINGS.md](BLOCK_BUDGET_FINDINGS.md) and [GROUND_CONTACT.md](GROUND_CONTACT.md)).

### Existing Full-Body Model Budgets

| Model Variant | Compiled Blocks | Headroom to 1,000 Ceiling | Headroom to 975 Reserve Gate |
| --- | ---: | ---: | ---: |
| `GS3DX_FullBody` | 945 | 55 | 30 |
| `GS3DX_FullBodyContact` | 967 | 33 | 8 |
| `GS3DX_Neck` / `GS3DX_Human` | **975** | **25** | **0** |

### Block Cost to Embed Physical Grip Contact Into `GS3DX_Human`

Replacing the welded hand–club loop in `GS3DX_Human` with the physical grip-contact subsystem requires the following compiled nonvirtual blocks:

| Change in `GS3DX_Human` | Block Type / Compiled Expansion | Compiled Block Delta |
| --- | --- | ---: |
| Delete `Right Hand Weld` (`Weld Joint`) | `SimscapeMultibodyBlock` | $-1$ |
| Add `Club 6DOF` (`6-DOF Joint` between `Clubface Vector` and `Club_Origin`) | `SimscapeMultibodyBlock` | $+1$ |
| Add `Club_Principal_Transform` (`Rigid Transform` for principal inertia axes) | `SimscapeMultibodyBlock` | $+1$ |
| Add 3 `Transform Sensor` blocks (`Club`, `Hand_L`, `Hand_R` vs. `World`) | 3 $\times$ `SimscapeMultibodyBlock` | $+3$ |
| Add 12 `PS-Simulink Converter` blocks ($p, R, v, \omega$ for 3 bodies) | 12 $\times$ (`PMIOPort` + `SimscapeRtp`) | $+12$ to $+24$ |
| Add `Grip Contact Kernel` (`MATLAB Function` block) | 1 $\times$ `S-Function` | $+1$ |
| Add `Slip State Integrator` + `Style_Code` + `Squeeze_N` (`Integrator` + 2 `Constant`) | 1 `Integrator` + 2 `Constant` | $+3$ |
| Add `Club External Wrench` (`External Force and Torque` on club) | `SimscapeMultibodyBlock` | $+1$ |
| Add 2 vector (or 6 scalar) `Simulink-PS Converter` blocks for club wrench | 2 to 6 $\times$ `PMIOPort` | $+2$ to $+6$ |
| Add bilateral hand reaction `External Force and Torque` (`Hand_L`, `Hand_R`) + converters | 2 `SimscapeMultibodyBlock` + 4 to 12 `PMIOPort` | $+6$ to $+14$ |
| **Total Incremental Compiled Blocks (Club-Only / Bilateral Hands)** | | **$+23$ to $+52$** |
| **Resulting `GS3DX_Human` Compiled Total** | $975 + (23\text{..}52)$ | **998 to 1,027** |

**Conclusion:** `GS3DX_Human` is already at **975 compiled blocks** (zero headroom under the 975 reserve gate and only 25 blocks below the hard 1,000-block Home-license ceiling). Even the minimal club-only physical grip contact loop adds at least **+23 compiled blocks** (bringing `GS3DX_Human` to $\ge 998$ blocks, violating the 975 reserve gate), and including bilateral hand reaction wrenches plus converter expansions brings the compiled total to **1,004–1,027 blocks**, failing compilation outright under the Home license. Therefore, per the issue #13334 design gate, physical grip contact is built as the standalone `GS3DX_GripContact` fixture model (~80–95 compiled blocks, $> 900$ blocks of headroom) rather than modifying `GS3DX_Human.slx`.

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
