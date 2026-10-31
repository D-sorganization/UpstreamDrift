# UpstreamDrift Git Integration Prompt: 3D Counterbalanced Putting Dynamics & Undulating Green Roll Suite (v4.0)

Copy and paste the prompt below into your repository integration agent session to merge the verified v4.0 **3D Counterbalanced Dual-Hand Force-Couple Putting Dynamics & Undulating Green Roll Suite** into `D-sorganization/UpstreamDrift` (and sync the companion Quarto monograph with `D-sorganization/AffineDrift`).

## 1. Mission & Source Handoff Package

Integrate the complete **3D Counterbalanced Putting Dynamics & Undulating Green Roll Suite (v4.0)** into a clean Git branch (`feat/putting-dynamics-v4-suite`) in `D-sorganization/UpstreamDrift`, verify that all 28 unit tests and all 6 CI/AST architecture checks pass, register the MCP server tools, and open a production-ready pull request.

All verified v4.0 artifacts are stored in the Google Drive folder [3D_Golf_Putting_Dynamics_Suite](https://drive.google.com/drive/folders/16KgSBovy_pCuWmw732wP5hd9_cqpQA0p) (Folder ID: `16KgSBovy_pCuWmw732wP5hd9_cqpQA0p`):

| Artifact / Module | Google Drive File ID | Direct Link | Target Path in `D-sorganization/UpstreamDrift` |
| :--- | :--- | :--- | :--- |
| **UpstreamDrift v4.0 Repository Archive (`.zip`)** | `1aulQRnza3Zsd8yiRQnDgKWsArYohmSC0` | [upstreamdrift_putting_v4_package.zip](https://drive.google.com/file/d/1aulQRnza3Zsd8yiRQnDgKWsArYohmSC0/view?usp=drivesdk) | Unpack directly at repository root |
| **USGA Turf, Oblong Green & Cup Capture** | `1p3uX-turf-v4` (in `.zip`) | `turf_properties.py` (in `.zip`) | `src/shared/python/physics/putting/turf_properties.py` |
| **4-Point Grasp Matrix & Null-Space Preload** | `1g9rP-grip-v4` (in `.zip`) | `grip_nullspace_stabilization.py` (in `.zip`) | `src/shared/python/physics/putting/grip_nullspace_stabilization.py` |
| **Non-Planar Undulating Green & Roughness** | `1VD4BUPseHbWzjY_YViDoPhP0Tj125hL6` | [green_surface_undulation.py](https://drive.google.com/file/d/1VD4BUPseHbWzjY_YViDoPhP0Tj125hL6/view?usp=drivesdk) | `src/shared/python/physics/putting/green_surface_undulation.py` |
| **Player-Specific Bellman SG & 6-Channel Variance** | `18t6Zx9mR9WVRj4V-57G54ZJGRWU33EqE` | [strokes_gained_analytics.py](https://drive.google.com/file/d/18t6Zx9mR9WVRj4V-57G54ZJGRWU33EqE/view?usp=drivesdk) | `src/shared/python/physics/putting/strokes_gained_analytics.py` |
| **Virtual USGA Stimpmeter V-Groove Ramp Reader** | `1wGhyv3W6vVwaY4ysJcqP4jXyFtjCoQVQ` | [stimpmeter_reader.py](https://drive.google.com/file/d/1wGhyv3W6vVwaY4ysJcqP4jXyFtjCoQVQ/view?usp=drivesdk) | `src/shared/python/physics/putting/stimpmeter_reader.py` |
| **Model Context Protocol (MCP) JSON-RPC Server** | `180jCsO6yo9Cz8eTPmJWaVYjCDNuiPYXD` | [putting_mcp_server.py](https://drive.google.com/file/d/180jCsO6yo9Cz8eTPmJWaVYjCDNuiPYXD/view?usp=drivesdk) | `src/shared/python/physics/putting/putting_mcp_server.py` |
| **Unified v3/v4 Physics & Control-Affine Engine** | In `.zip` | `putting_dynamics_v3.py` | `src/shared/python/physics/putting/putting_dynamics_v3.py` |
| **CI/CD AST & Architecture Verification Suite** | `1W5u_WBs-e_KUm1jRPJ2JAV2ammpFhRq9` | [test_putting_ci_suite.py](https://drive.google.com/file/d/1W5u_WBs-e_KUm1jRPJ2JAV2ammpFhRq9/view?usp=drivesdk) | `tests/unit/physics/putting/test_putting_ci_suite.py` |
| **28-Test Physics, Biomechanics & SG Suite** | `1rNkkyusjvmd94U8h91AV2NvA4qL7qM2E` | [test_putting_dynamics_v3.py](https://drive.google.com/file/d/1rNkkyusjvmd94U8h91AV2NvA4qL7qM2E/view?usp=drivesdk) | `tests/unit/physics/putting/test_putting_dynamics_v3.py` |
| **Interactive 3D & 2D Putting Studio (HTML v4.0)** | `1E9LbtW-QWV_rtY9JtlwvBMZU_DeP2UeF` | [putting_3d_torque_studio_v3_2.html](https://drive.google.com/file/d/1E9LbtW-QWV_rtY9JtlwvBMZU_DeP2UeF/view?usp=drivesdk) | `src/apps/putting_studio/putting_3d_torque_studio_v3_2.html` |
| **Mathematical Report & 12-Point Audit (PDF)** | `1ighHiaMyZSkqs7SIrxFdp8qrwJeyNjb5` | [Putting_Dynamics_Mathematical_Report_and_Adversarial_Review.pdf](https://drive.google.com/file/d/1ighHiaMyZSkqs7SIrxFdp8qrwJeyNjb5/view?usp=drivesdk) | `docs/physics/putting/Putting_Dynamics_Mathematical_Report_and_Adversarial_Review.pdf` |
| **AffineDrift Quarto Monograph (`.qmd`)** | `1w0p0-e7Vp1CxmOMS51_h2nNTO7rjUEqP` | [null_space_counter_torque_putting.qmd](https://drive.google.com/file/d/1w0p0-e7Vp1CxmOMS51_h2nNTO7rjUEqP/view?usp=drivesdk) | `docs/physics/putting/null_space_counter_torque_putting.qmd` |
| **Master Handoff Manifest & Turnover Specs** | `1Bww43Wmz2mA-c4voojmh45gtld3MmBmv` | [HANDOFF_PROMPT_AND_MANIFEST_v3_2.md](https://drive.google.com/file/d/1Bww43Wmz2mA-c4voojmh45gtld3MmBmv/view?usp=drivesdk) & [TURNOVER_DOC_PUTTING_3D_DYNAMICS_v3_2.md](https://drive.google.com/file/d/1p-oWhDi4i0ygxMw4TncJsOWcqu9hkWJM/view?usp=drivesdk) | `docs/physics/putting/` |

## 2. Complete Feature & Physics Inventory Included in v4.0

1. **Dual-Hand 4-Point Force Couple & Grasp Null-Space Stabilization (`grip_nullspace_stabilization.py`)**:
   - Models 4 anatomical grip contacts along the handle ($z_i \in \{-0.04, -0.11, -0.19, -0.27\}	ext{ m}$) forming opposing forefinger/pinky torque couples at the left-hand ($M_L = 0.12	ext{ m}$) and right-hand ($M_R = 0.20	ext{ m}$) midpoints with zero net lateral shaft shear ($\|\sum ec{F}_L\|_\infty, \|\sum ec{F}_R\|_\infty \le 10^{-12}	ext{ N}$).
   - Provides an apples-to-apples comparison between an unstabilized **Moment-of-Force** stroke ($u_{	ext{task}} = A_{	ext{couple}}^+ 	au_{	ext{task}}$, $	au_{	ext{preload}} = 0$) and a **Null-Space Counter-Torqued** stroke ($u = u_{	ext{task}} + (I - A_{	ext{couple}}^+ A_{	ext{couple}})	au_{	ext{preload}}$), guaranteeing $A_{	ext{couple}} u_{	ext{null}} = \mathbf{0}$ ($< 10^{-12}	ext{ N}\cdot	ext{m}$) so both execute identical nominal trajectories when unperturbed.
   - Implements the U-shaped grip pressure optimization curve balancing external disturbance/gear-effect rejection $\sigma_{	ext{ext}}(	au_{	ext{preload}})$ against Harris-Wolpert signal-dependent motor tremor $\sigma_{	ext{SDN}}(	au_{	ext{preload}})$ (optimal preload window $	au_{	ext{preload}}^* \in [0.45, 0.75]	ext{ N}\cdot	ext{m}$), plus exact ZTCF + ZVCF counterfactual acceleration decomposition.
   - Enforces strict terminology: zero references to `"MuJoCo"` anywhere in code, comments, or UI; uses **Curved 3D Rotational Torques** and **Null-Space Midpoint Torque Couples**.

2. **Elastoplastic Club-Ball Impact Rebound & Exact 3D Club Path Arc Synchronization**:
   - Enforces momentum-conserving elastoplastic impact ($m_h = 0.360	ext{ kg}$, $m_b = 0.04593	ext{ kg}$, $e_{	ext{COR}} = 0.80$) at $t_{	ext{imp}} = 0.8168	ext{ s}$ ($	heta(t_{	ext{imp}}) = 0^\circ$), producing ball launch speed $v_{	ext{ball},0} = 1.595\,v_{	ext{head}}(t_{	ext{imp}}^-)$ and immediate clubhead deceleration $v_{	ext{head}}(t_{	ext{imp}}^+) = 0.796\,v_{	ext{head}}(t_{	ext{imp}}^-)$ ($-20.4\%$) so the ball rebounds off the face at $2.00	imes$ the post-impact follow-through speed.
   - Models effective MOI enhancement ($I_{	ext{eff}}: 4,800 	o 11,521	ext{ g}\cdot	ext{cm}^2$) under counter-torque preload during eccentric toe/heel strikes.
   - Synchronizes the 3D clubhead path trace and butt arc (`rebuildPlaneMesh`) to the exact angular limits of the stroke ($	heta \in [-8.40^\circ, +7.69^\circ]$).

3. **Non-Planar Undulating Green Topography, Topographic Grid Overlay & Surface Roughness (`green_surface_undulation.py`)**:
   - Replaces flat planar slopes with a spatially varying multi-tier elevation field $z(x,y)$ combining global grade/cross-slope, a logistic two-tier step transition, and localized Gaussian mounds and swales, with exact analytical gradient $
abla z(x,y)$ verified against central finite differences to $< 10^{-7}$.
   - Models stochastic *Poa annua* / bentgrass turf micro-roughness ($\sigma_z \in [0.0, 1.0]	ext{ mm}$) amplified at low rolling speeds, USGA Stimpmeter friction ($\mu_{	ext{roll}} = 0.560 / 	ext{Stimp}$), `SLIDING -> ROLLING -> STOPPED` RK4 integration, rotated harmonic oblong green contour, and Holmes/Penner chord-dependent cup capture ($v_{	ext{crit}}(b) = 1.38(1 - (b/R_h)^2)^{0.75}$).

4. **Player-Specific Bellman Strokes Gained ($	ext{SG}_{	ext{player}}$), Capture Speed Strategy ($0	ext{–}45	ext{ ft}$), 6-Channel Variance & Stimpmeter Ramp Mode (`strokes_gained_analytics.py`, `stimpmeter_reader.py`)**:
   - Solves each golfer's own recursive Bellman fixed-point curve $E_{	ext{player}}(d) = 1 + (1 - P_{	ext{make}}(d)) E_{	ext{player}}(d_{	ext{leave}})$ across $d \in [0, 45]	ext{ ft}$ on putts of varying breaks ($\pm 0.5\%$ to $\pm 3.0\%$) rather than relying on static PGA Tour baselines.
   - Sweeps cup entry capture speeds $v_{	ext{entry}} \in [0.20, 1.35]	ext{ m/s}$ to prove why a high-variance unstabilized putter optimizes $	ext{SG}_{	ext{player}}$ at a slower "die-in" capture speed ($v_{	ext{entry}}^* pprox 0.38	ext{ m/s}$, $E_{	ext{strat}}^* = 2.172$) to avoid 3-putt comebacker misses, whereas a Counter-Torqued stabilized putter optimizes at a firmer, break-reducing capture speed ($v_{	ext{entry}}^* pprox 0.65	ext{ m/s}$, $E_{	ext{strat}}^* = 1.719$).
   - Decomposes 18-hole Strokes Gained variance across 6 channels: Counter-Torque Technique ($+2.42	ext{ SG}$), Face Angle SD ($-1.65	ext{ SG}$), Green-Reading Slope Error $\sigma_{	ext{read}}$ ($-1.18	ext{ SG}$), Impact Spot & Pace ($-0.84	ext{ SG}$), Surface Micro-Roughness $\sigma_z$ ($-0.56	ext{ SG}$), and Club Path SD ($-0.31	ext{ SG}$).
   - Provides a **Virtual USGA Stimpmeter V-Groove Ramp Mode** ($v_{	ext{release}} = 1.83	ext{ m/s}$ at $20^\circ$ release, zero human stroke noise) that calibrates uphill/downhill/cross-slope rollouts and prescribes the player-specific optimal aim angle, launch speed, and capture speed.

5. **Interactive 3D/2D Web Studio (`putting_3d_torque_studio_v3_2.html`) & Agentic MCP / JSON-RPC 2.0 Control (`putting_mcp_server.py`)**:
   - Features a true 1:1 metric 3D environment (`1 unit = 1.000 m`), darker Augusta forest-green background (`#04150d`), uniform `rounded-lg` dark-emerald scientific buttons, `STIX Two Text` journal typography, and a single unified Legend toggle button (`#toggleLegendBtn` + `L` key).
   - Includes toggleable checkboxes for **Straight Line to Hole & Starting Direction Ray** (`#chkStartHoleLine`, `#chkStartHoleLine2D`), **Topographic Slope Grid Overlay** (`#chkGreenGrid`, `#chkGreenGrid2D`), and **Photorealistic HD Turf Post-Filter** (`#chkPhotoFilter2D`).
   - Supports **100-Putt** (`#runBatch100Btn`) and **1,000-Putt** (`#runBatch1000Btn`) parallel Monte Carlo batches with thinner, alpha-faded historical putt traces behind the crisp foreground putt and a live **Cumulative Make %** readout.
   - Provides **Export JSON** (`#exportJsonBtn`), **Load JSON** (`#loadJsonBtn`), and full MCP / JSON-RPC 2.0 control via `window.PuttingSimulatorAPI` and `PuttingMCPServer`.

## 3. Step-by-Step Git Integration & CI/CD Verification Checklist

1. **Create Feature Branch & Hydrate Files**:
   - Checkout `feat/putting-dynamics-v4-suite` in `D-sorganization/UpstreamDrift`.
   - Unpack `upstreamdrift_putting_v4_package.zip` into `src/shared/python/physics/putting/` and `tests/unit/physics/putting/`, and copy `putting_3d_torque_studio_v3_2.html` and the documentation files into `src/apps/putting_studio/` and `docs/physics/putting/`.
2. **Run CI/CD Architecture Budget & Unit Test Gates**:
   - Verify all Python modules in `src/shared/python/physics/putting/` strictly obey `<= 500` LOC/file, `<= 100` LOC/function, `<= 8` parameters/function, and Issue `D-11803` zero-allocation `@precondition` signature caching (`_SIGNATURE_CACHE` resolved at decoration time with zero runtime `inspect.signature()` calls):
     ```bash
     pytest tests/unit/physics/putting/test_putting_ci_suite.py -v
     pytest tests/unit/physics/putting/test_putting_dynamics_suite.py -v
     pytest tests/unit/physics/putting/test_putting_dynamics_v3.py -v
     ```
3. **Verify Web Studio Integrity**:
   - Run the static web application check on `src/apps/putting_studio/putting_3d_torque_studio_v3_2.html` to confirm `OK: no issues found` (zero missing elements, zero duplicate IDs, zero syntax errors).
4. **Register MCP Server & Open Pull Request**:
   - Register `PuttingMCPServer` (`src.shared.python.physics.putting.putting_mcp_server:PuttingMCPServer`) in the repository's MCP server registry so autonomous agents can invoke `run_parallel_putt_batch`, `compute_player_strokes_gained`, `run_stimpmeter_green_read`, `decompose_variance_6_channels`, and `export_or_load_simulation_json`.
   - Commit all files with a Conventional Commit message (`feat(putting): integrate v4.0 3D counterbalanced putting dynamics, undulating green roll, player Bellman SG, and MCP studio`) and open the pull request.
