# Argon Deep Technical Audit: `UpstreamDrift` & `Tools`

## Executive Summary

This document records the Phase 1 file-and-line technical audit across `D-sorganization/UpstreamDrift` and `D-sorganization/Tools` (`vendor/ud-tools`) conducted by the Argon Lead Architect. It covers the five core architectural and scientific integrity vectors and establishes the foundation for the 5 Master Epics (`#9700`, `#11605`, `#11987`, `#9406`, and `#9415`).

---

## Vector 1: Physics & Energy Audit Integrity (`src/`, `rust_core/`, & `vendor/ud-tools`)

### 1.1 Impact Interval & Energy Conservation Calculations

- **Provider Implementation (`Tools/src/shared/python/swing_sim/impact_interval/solver.py`):**
  - _Lines 266–372:_ Integrates contact mechanics over time steps up to `max_steps = floor(maximum_time_s / dt) + 1`, computing normal contact force, regularized tangential friction, and stored spring energy `stored_contact = 0.5 * k * d^2`.
  - _Lines 273–278:_ Tracks tensile clipping via `unilateral_release += max(0.0, stored_contact_previous - stored_contact)` when `tensile_clip_pending` is active. Historically (prior to Tools PR #5079), `unilateral_release = max(0.0, raw_energy_residual)` dumped numerical drift and truncated-contact energy deficits (e.g., 5.777 J on a truncated probe) into `unilateral_release` to force a `0.0 J` residual.
  - _Lines 391–398:_ Computes the signed energy residual `energy_residual_j`.
- **Audit Consumer (`src/shared/python/physics/impact_interval_audit.py`):**
  - _Lines 145–159 (`recompute_energy_residual`):_ Independently recomputes the signed energy residual from ledger terms.
  - _Lines 162–252 (`audit_impact_interval`):_ Verifies contact completion/separation, `|energy_residual_j| <= tolerance`, and linear/angular momentum closure.
  - _Lines 255–280 (`qualified_post_impact_state`):_ Rejects exporting post-impact states when contact is unfinished (`#9547`) or energy is unclosed.
- **Rust Core Contact Kernel (`rust_core/upstream-physics/src/contact.rs`):**
  - _Lines 111–186 (`calculate_impact`):_ Computes pre- and post-impact kinetic energy `ke_in` and `ke_out`.
  - _Line 183:_ Clamps `energy_lost: energy_lost.max(0.0)`, which silently zeroes negative energy loss (non-conservative energy creation from numerical discretization or tangential overshoot).
- **MuJoCo Energy Monitor (`src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/verification.py`):**
  - _Lines 35–140 (`EnergyMonitor`):_ Approximates mechanical work via explicit Euler `power = np.dot(control_torques, self.data.qvel)` with a `1.0 J` default tolerance.

### 1.2 Usage of `GOLF_USE_MOCK_ENGINE`, Stubs, and CI Fallbacks

- **`MockPhysicsEngine` (`src/shared/python/engine_core/mock_engine.py:30-100`):**
  - Activated when `GOLF_USE_MOCK_ENGINE=1`. Advances positions via unconstrained Euler stepping (`self._positions += self._velocities * dt`).
  - Used in `tests/unit/test_energy_monitor.py:77-415`, `tests/unit/test_dataset_generator.py:32`, and `tests/biomechanics/test_recording_integration.py:19`.
- **Static Typing Stubs vs Runtime Mocks (`stubs/pinocchio/__init__.pyi` & `tests/conftest.py`):**
  - `stubs/pinocchio/__init__.pyi` provides `mypy` type hints for `Model`, `Data`, `SE3`, `Motion`, and `buildModelFromUrdf`.
  - `tests/conftest.py:1110-1175` defines `mock_drake_dependencies` and `mock_mujoco_dependencies` fixtures that inject `MagicMock` into `sys.modules`.
  - `src/shared/python/engine_core/engine_availability.py:115-179` guards against mock pollution by raising `ImportError("mocked pinocchio module detected")` when `type(pin).__module__ == "unittest.mock"`.

---

## Vector 2: Cross-Engine Adapter Contracts (`src/engines/`)

### 2.1 Engine Registry & Protocol Boundaries

- **Tier Definitions (`src/engines/tiers.py:9-17`):**
  - Currently registers `mujoco` and `putting_green` (`core`), `drake` and `pinocchio` (`extended`), and `opensim`, `myosuite`, and `jaxsim` (`experimental`).
  - **Gap:** `simscape` is absent from `ENGINE_TIERS`, and `tiers.py` classifies installation footprint rather than mathematical equivalency tiers.
- **Protocol Contract (`src/engines/protocols.py:12-77`):**
  - `PhysicsEngineProtocol` models rigid-body generalized coordinates $q$, velocities $\dot{q}$, and joint torques $ au$. It does not natively express muscle excitations $u$, activations $a$, fiber lengths $l_m$, or tendon dynamics.

### 2.2 Adapter Audit & the 3-Tier Equivalency Contract

- **OpenSim (`src/engines/physics_engines/opensim/python/full_body_parity.py:17-20, 62-67`):**
  - Strips `updMarkerSet`, `updForceSet`, `updConstraintSet`, and `updContactGeometrySet` to run as a pure Simbody rigid-body tree under external KKT contact for Tier 1 comparison (`#11605`), while native muscle/contact qualification is governed separately under `#11784` (`F07`/`F08`) and `Tools#5460`.
- **MyoSuite (`src/engines/physics_engines/myosuite/python/full_body_parity.py:48-55`):**
  - `MyoSuiteFullBodyModel` subclasses `NativeMujocoFullBodyModel` for Tier 1 rigid-body sanity checks, whereas native muscle-excitation replay is tracked under `#10346`, `#11642`, and `#11784`.
- **Simscape Multibody (`src/engines/simscape/adapter.py` & `#12013`):**
  - `src/engines/simscape/adapter.py:718` (`compute_contact_forces`) returns a zero array `np.zeros(6)` when contact forces are absent in `GolfSwing3D_Kinetic`, violating the `GCV-3` (`#11709`) requirement that unavailable channels return `None` / unavailable rather than fabricated zeros.
  - `tests/fixtures/cross_engine_parity/simscape_canonical_baseline.npz` (`#12013`) is quarantined as `withheld-not-validated-as-simscape` in `tests/cross_engine/baseline_provenance.py:61-76`.

---

## Vector 3: Dual-UI Split-Brain (`ui/`, `src/api/`, and PyQt6 `--classic`)

### 3.1 Launcher & Parity Registry (`launch_upstream_drift.py` & `src/config/feature_parity.json`)

- `launch_upstream_drift.py` launches the FastAPI + React/Tauri web UI by default (`src.api.local_server.main()`) and provides `--classic` for the PyQt6 desktop shell (`src.launchers.upstream_drift_launcher.main()`).
- `src/config/feature_parity.json` tracks 62 features: 39 `parity`, 7 `gap`, and 16 `exempt` desktop-only tools.

### 3.2 Exact Delta List of Active Web Parity Gaps

1. **`tools.matched_swing_browser` (`#11987`):** `ui/src/pages/MatchedSwings.tsx` supports ledger browsing, receipts, and 3D preview, but lacks a button wired to `POST /api/matched-swings/{run_id}/native-viewer` (`src/api/routes/matched_swings_native_viewer.py`).
2. **`tools.launch_monitor_analytics` (`#11987`):** `ui/src/pages/LaunchMonitorAnalytics.tsx` implements flexible analysis, trends, dispersion, relationships, and model fitting, but omits the Sessions and Reports tabs exposed by `src/api/routes/launch_monitor_analytics_sessions.py` and `src/api/routes/launch_monitor_analytics_reports.py`.
3. **`tools.rate_of_closure` (`#11987`):** `ui/src/pages/ImpactExplorer.tsx` embeds `/impact-explorer-app/`; `src/config/impact_zone_readiness.json` contains stale status entries for `#9349`.
4. **`tools.bunkershot3d_workbench` (`#9545`):** `src/api/routes/bunker_workbench.py` provides the v1 solve endpoint (with PR `#12449` in flight for A/B comparison), awaiting dedicated React page integration.
5. **`lifting.five_lift_viewing` (`#11748`):** `ui/src/pages/LiftBaseline.tsx` renders start-pose tables and parameters; barbell/hand-force 3D overlays are tracked under Epic `#11740`.
6. **`render.club_head_and_ball` (`#11719`):** `ui/src/components/visualization/ClubHead.tsx` renders STL heads; decorative ball and trajectory trace in `GolferModel.tsx` are tracked under `#11719`.
7. **`tools.model_explorer.frankenstein_assembly` (`#11651`):** `src/api/routes/model_explorer_assembly.py` provides the stateless REST API; web canvas parity is tracked under Epic `#11651`.

---

## Vector 4: Submodule & Packaging Fragility (`vendor/ud-tools` & `pyproject.toml`)

### 4.1 Shadow Tree vs Pinned `ud-tools` Dependency (`#9406`)

- `src/shared/python/` (5,215 files) and `vendor/ud-tools/src/shared/python/` (3,921 files) share 1,137 relative paths.
- Per `docs/shared_tools/seam_rulings.v1.json` and Epic `#9406`, `D-sorganization/Tools` (`ud-tools`) is the canonical upstream provider pinned identically across `.gitmodules`, `requirements-tools.txt`, and `Cargo.toml`.
- Windows native test processes mixing `opensim` (bundled `ezc3d.dll`) and Python `ezc3d` require process-boundary isolation via source-bound TRC receipts (`#12065`, PR `#12451` and PR `#12454`).

### 4.2 Optional Engine Extras (`pyproject.toml` & `Makefile`)

- `pyproject.toml` currently places `mujoco` in core dependencies, groups `opensim` and `myosuite` under `biomechanics`, and lacks individual `mujoco`, `opensim`, `myosuite`, and `ui` optional dependency keys.
- `launch_upstream_drift.py:46` references `pip install "upstream-drift[tools]"`, which does not exist in `pyproject.toml` (moved to `requirements-tools.txt` for PEP 508 compliance).
- `Makefile` (`install` target) runs `pip install -e ".[dev]"` without installing `-r requirements-tools.txt`.

---

## Vector 5: Root Directory & Agent Config Sprawl

### 5.1 Root Clutter Gate Loophole (`scripts/check_root_clutter.py`)

- `scripts/check_root_clutter.py:57-59` skips all directories (`if entry.is_dir() or entry.name.startswith("."): continue`), allowing arbitrary top-level directories to bypass CI checks.
- Untracked local debris in the primary checkout (`UsersdieteRepositorieswt-11c/`, `sessions/`, `worktrees/`, `logs/`, and root `*.py`/`*.json` scratch scripts) was identified and requires root-directory allowlist enforcement and `.gitignore` coverage.

### 5.2 Hidden Agent Folders & Governance

- Tracked agent configuration directories (`.agent`, `.claude`, `.gaai`, `.gemini`, `.jules`, `.kiro`, `.specify`) coexist at the repository root. A unified `.ai/README.md` governance index is established to document their roles while preserving `AGENTS.md` and `CLAUDE.md` as the authoritative fleet entry points.
