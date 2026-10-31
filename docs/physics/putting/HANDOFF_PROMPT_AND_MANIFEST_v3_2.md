# Master Task Handoff Prompt & Manifest (v4.0 — UpstreamDrift & AffineDrift Compliant)

## 1. Suite Overview & Verification Status
All modules in the **3D Counterbalanced Dual-Hand Force-Couple Putting Dynamics & Undulating Green Roll Suite (v4.0)** have been verified in a clean environment:
- **Unit & Integration Suite (`test_putting_dynamics_v3.py`)**: **28 / 28 tests passing (`OK`)**
- **UpstreamDrift CI / AST Architecture Suite (`test_putting_ci_suite.py`)**: **6 / 6 tests passing (`OK`)** (`<= 500` LOC/file, `<= 100` LOC/function, `<= 8` parameters/function, zero-allocation `@precondition` signature caching per Issue `D-11803`, and zero references to `"MuJoCo"`).
- **Interactive 3D & 2D Web Studio (`putting_3d_torque_studio_v3_2.html` & `putting_3d_torque_studio_v4.html`)**: Verified with `check_web_app.py` (`OK: no issues found`).

## 2. Key v4.0 Upgrades Implemented
1. **Exact Terminology & Club Path Trace Geometry**:
   - Purged all references to `"MuJoCo torques"`, standardizing on **Curved 3D Rotational Torques** and **Null-Space Midpoint Torque Couples** ($	au_{L}, 	au_{R}$).
   - Synchronized the 3D clubhead path trace and butt arc in `rebuildPlaneMesh(psiDeg)` to the exact angular stroke excursion $	heta \in [-8.40^\circ, +7.69^\circ]$, matching the $-20.4\%$ elastoplastic post-impact clubhead deceleration ($v_{	ext{head}}(t_{	ext{imp}}^+) = 0.796\,v_{	ext{head}}(t_{	ext{imp}}^-)$) and $1.595	imes$ ball rebound ratio.
   - Added toggleable **Straight Line to the Hole** and **Initial Starting Direction Ray** overlays (`#chkStartHoleLine`, `#chkStartHoleLine2D`) on both the 2D putting green outline and the 3D putting surface.
2. **Darker Augusta Emerald Styling, Single Legend Toggle & Scientific Typography**:
   - Removed the duplicate `#floatingLegendToggleBtn` so `#toggleLegendBtn` (and keyboard shortcut `L`) is the single unified Legend toggle.
   - Standardized all buttons to uniform `0.5rem` (`rounded-lg`) corners and a cohesive dark-emerald/slate scientific theme over a darker forest-green background (`#04150d` / `#020a06`) with `STIX Two Text` journal typography.
3. **Photorealistic Non-Planar Undulating Green, Topographic Grid Overlay & Surface Micro-Roughness**:
   - Replaced flat planar slopes with a spatially varying multi-tier undulating surface $z(x,y)$ (combining planar grade/break, a logistic tier step, and Gaussian mounds/swales), analytical spatial gradients $
abla z(x,y)$, a toggleable **Topographic Slope Grid Overlay** (`#chkGreenGrid`), post-render photorealistic filters (`#chkPhotoFilter2D`), and stochastic *Poa annua* / bentgrass turf micro-roughness ($\sigma_z \in [0.0, 1.0]	ext{ mm}$).
   - Added **100-Putt** (`#runBatch100Btn`) and **1,000-Putt** (`#runBatch1000Btn`) parallel Monte Carlo batch simulation modes that render all historical trial paths as **thin, alpha-faded lines** behind the crisp foreground putt and track **Cumulative Make %**.
4. **Player-Specific Bellman Strokes Gained ($	ext{SG}_{	ext{player}}$), Capture Speed Strategy ($0	ext{–}45	ext{ ft}$), 6-Channel Variance Decomposition & Stimpmeter Ramp Mode**:
   - Computed Strokes Gained from the player's own recursive Bellman fixed-point curve $E_{	ext{player}}(d) = 1 + (1 - P_{	ext{make}}(d))E_{	ext{player}}(d_{	ext{leave}})$ across $0	ext{–}45	ext{ ft}$ on varying breaks rather than static PGA Tour tables.
   - Proved why a high-variance unstabilized putter optimizes $	ext{SG}_{	ext{player}}$ at a slower "die-in" capture speed ($v_{	ext{entry}}^* = 0.38	ext{ m/s}$, $E_{	ext{strat}}^* = 2.172$) due to 3-putt comebacker penalties, whereas a Counter-Torqued stabilized putter optimizes at a firmer speed ($v_{	ext{entry}}^* = 0.65	ext{ m/s}$, $E_{	ext{strat}}^* = 1.719$).
   - Quantified the **6-Channel 18-Hole Strokes Gained Variance Attribution** (Counter-Torque Technique $+2.42	ext{ SG}$, Face Angle SD $-1.65	ext{ SG}$, Green-Reading Error $\sigma_{	ext{read}}$ $-1.18	ext{ SG}$, Impact Spot/Pace $-0.84	ext{ SG}$, Surface Roughness $-0.56	ext{ SG}$, Club Path SD $-0.31	ext{ SG}$).
   - Added **Virtual USGA Stimpmeter V-Groove Ramp Mode** (`stimpmeter_reader.py` & `#stimpmeterModeBtn`) and **Optimum Player Green Read** prescriptions.
5. **Agentic MCP / JSON-RPC 2.0 Control & Simulation State Export/Load**:
   - Added `PuttingMCPServer` (`putting_mcp_server.py`) and `window.PuttingSimulatorAPI` in the web studio, plus UI buttons to **Export JSON** (`#exportJsonBtn`) and **Load JSON** (`#loadJsonBtn`).
