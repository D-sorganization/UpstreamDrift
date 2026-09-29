# UI/UX, Performance, and Proximal–Distal Modeling Review — 2026-08-21

Second review wave requested by the repository owner, following the
integration/organization review (PR #8815, issues #8816–#8876). Three
axes this wave, each verified against the code before filing:

1. **UI/UX** — desktop launcher and shared UI infrastructure; tool GUIs
   and the React/Tauri web app. Responsiveness, error-state UX,
   feedback, consistency, accessibility, state persistence.
2. **Performance** — runtime hot paths (vectorization, per-frame
   allocation, plotting), startup cost, API latency, caching, IPC, and
   Rust-acceleration opportunities.
3. **Proximal–distal modeling** — critical technical review of the
   energy-transfer research program (epics #8557/#8668/#8684):
   assumptions, energy accounting, friction modeling, cross-engine
   parity, experimental design, claim governance.

Dedup notes: `.scratch/uiux-issues/` holds 28 pre-drafted web-app UI/UX
issues; at least drafts 01–03 were filed as #7415–#7417 (closed). New
findings were checked against those drafts and against open issues
before filing.

## Findings and Filed Issues

_(Populated as the review proceeds; each row is a filed GitHub issue.)_

| #   | Issue | Area        | Severity | Title                                                                                              |
| --- | ----- | ----------- | -------- | -------------------------------------------------------------------------------------------------- |
| 1   | #8879 | UX tools    | Critical | Pose Studio crashes on construction (act_undo used before create_menu_bar)                         |
| 2   | #8880 | UX tools    | High     | Every simulation runs on the GUI thread; no progress, no cancel                                    |
| 3   | #8881 | UX tools    | High     | Launch Monitor New Project destroys unsaved work; dirty flag reset defeats close guard             |
| 4   | #8882 | UX tools    | High     | Pose Studio Save/Load are tooltip stubs; is_dirty hardcoded False; no closeEvent                   |
| 5   | #8883 | UX tools    | High     | Video Analyzer tile opens a placeholder label but is advertised ready                              |
| 6   | #8884 | UX tools    | High     | Training Controller swallows cancel/pause/resume failures; no cancel confirm                       |
| 7   | #8885 | UX tools    | Med-High | 45 hardcoded stylesheets bypass the theme; four colors for the run button                          |
| 8   | #8886 | UX tools    | Medium   | Units inconsistent across tools and mixed in result panes                                          |
| 9   | #8887 | UX tools    | Medium   | Pose Studio ±180° limits regardless of engine; silent edit rejection                               |
| 10  | #8888 | UX tools    | Medium   | Model Explorer per-engine exports write identical files                                            |
| 11  | #8889 | UX tools    | Medium   | Pose Matcher Clear overrides: no confirm + stale events without xlsx path                          |
| 12  | #8890 | UX tools    | Medium   | Terrain Engine/Model Explorer raise bare exceptions from Qt slots                                  |
| 13  | #8891 | UX web      | Medium   | Cross-Engine dashboard accepts dt=0; polls forever with no cancel                                  |
| 14  | #8892 | UX web      | Medium   | Web Settings discards unsaved edits on navigation                                                  |
| 15  | #8893 | UX tools    | Low-Med  | Golf Environment unlabeled mock trajectory; embed drops selector                                   |
| 16  | #8894 | UX launcher | Critical | Shared theme import broken; THEME_AVAILABLE always False in 5 widgets                              |
| 17  | #8895 | UX launcher | High     | Docker dialog Close bypasses build-cancel guard; untimed wait() freeze                             |
| 18  | #8896 | UX launcher | High     | Settings mixes live-apply and Apply-required tabs; Close discards edits                            |
| 19  | #8897 | UX launcher | High     | Layout: Locked caption never updates; Edit Tiles disabled without reason                           |
| 20  | #8898 | UX launcher | High     | Check Windows Dependencies imports pydrake in the clicked slot (5-15 s freeze)                     |
| 21  | #8899 | UX launcher | High     | Embedded-host workspace never persisted; state_snapshot/restore_state dead                         |
| 22  | #8900 | UX launcher | High     | Toasts unbounded, undismissable, window-detached, color-only type coding                           |
| 23  | #8901 | UX launcher | High     | Tile Favourite/Info hover-only 18px NoFocus; arrow-key nav documented but absent                   |
| 24  | #8902 | UX launcher | Med-High | Two shortcut dialogs both wrong (unbound list vs "(shortcut)" labels)                              |
| 25  | #8903 | UX launcher | Med-High | WSL checkbox and engine probes run blocking subprocesses on GUI thread                             |
| 26  | #8904 | UX launcher | Medium   | Integrations Health: failure=empty, sticky Copied!, no empty state, contrast                       |
| 27  | #8905 | UX launcher | Medium   | Empty filter results blank; per-keystroke full grid rebuild with smooth rescale                    |
| 28  | #8906 | UX launcher | Medium   | Skeleton cards animate nothing; forever timers                                                     |
| 29  | #8907 | UX launcher | Medium   | Five settings stores incl. dead branding; no window geometry persistence                           |
| 30  | #8908 | UX launcher | Low-Med  | Batch: unnamed modal, dark table in QMessageBox, global Esc, jargon, overlay resize, processEvents |
| 31  | #8909 | Research    | Critical | Friction-atlas MuJoCo–Pinocchio parity is MuJoCo vs itself (PyPI pinocchio 0.1; exact-zero errors) |
| 32  | #8910 | Research    | Critical | #8752 manufactured-solution controls are tautologies with hardcoded 0.0 conservation fields        |
| 33  | #8911 | Research    | High     | Cross-engine parity never exercises contact/constraint solvers or integrators                      |
| 34  | #8912 | Research    | High     | Unactuated 50 ms free response through ~100×-too-soft grip; "delivery proxy" outruns regime        |
| 35  | #8913 | Research    | High     | 384/576 cells are ~16× multiplicities of ~24 conditions; sign tally pools nested time steps        |
| 36  | #8914 | Research    | High     | Energy gates admit 2-5% residual with max(1.0,·) floors; effect never compared to dissipation      |
| 37  | #8915 | Research    | High     | No stick state: is_slipping is a velocity threshold; cone anchored to fiber tension                |
| 38  | #8916 | Research    | Med-High | 0/384 ground screen may be a denominator-floor artifact (1.0 N vs 1e-6 J)                          |
| 39  | #8917 | Research    | Med-High | 1-DOF wrist + discarded de Leva radii never bounded; touches the release mechanism                 |
| 40  | #8918 | Research    | Med-High | Claim governance verifies integrity, not numbers; register/attest loop is circular                 |
| 41  | #8919 | Research    | Medium   | pin unpinned + PyPI name collision; silent-success failure mode; runtimes unrecorded               |
| 42  | #8920 | Research    | Medium   | Document #8556-conditional parameters (k_grip, μ anchor, tension-only unilaterality)               |
| 43  | #8921 | Performance | Critical | Geometric IK: finite-difference Jacobian + per-call topology recompute (~10M FK/trial)             |
| 44  | #8922 | Performance | Critical | MuJoCo retargeting: mj_forward per marker in the IK loop (40× overcount)                           |
| 45  | #8923 | Performance | High     | Kalman smoother: 750k-iteration Python loop with provably constant gain                            |
| 46  | #8924 | Performance | High     | SciPy filters/resamplers called per (marker, axis) slice instead of axis=0                         |
| 47  | #8925 | Performance | High     | Normalization: per-marker matmul + Pydantic construction, four passes, no array path               |
| 48  | #8926 | Performance | High     | MarkerTrajectory validation: dead O(N·M) loop + sort-to-check-monotonic                            |
| 49  | #8927 | Performance | High     | Rust linear/cubic gap-fill kernels have zero Python callers; default stays Python                  |
| 50  | #8928 | Performance | High     | Simulation-result accessors recompute dynamics per index per series, uncached                      |
| 51  | #8929 | Performance | High     | Pendulum GUI: full dynamics in paintEvent at 60 fps; trail re-splined per repaint                  |
| 52  | #8930 | Performance | Med-High | Ball flight: scalar force path per point despite batch path; Python RK4 + 30-sin turbulence        |
| 53  | #8931 | Performance | Medium   | Recorder inverts the mass matrix per step; 5 factorizations of the same M                          |
| 54  | #8932 | Performance | Medium   | Analysis tabs recompute CWT and rebuild 3D axes per spinbox tick, blocking draw()                  |
| 55  | #8933 | Performance | Low-Med  | Batch: ezc3d Pydantic-per-point, recurrence loops, iterrows, buffer-growth shape bug               |
| 56  | #8934 | Performance | Critical | Engine discovery deep-imports pydrake.all/jaxsim/etc. on every start (est. 8-25 s)                 |
| 57  | #8935 | Performance | Critical | POST /simulate rebuilds the engine and re-parses the model every request                           |
| 58  | #8936 | Performance | Critical | WS sim loop steps physics on the event loop + wait_for(1ms) per step                               |
| 59  | #8937 | Performance | High     | Launcher manifest spawns 40-80 git subprocesses per request, uncached, on the event loop           |
| 60  | #8938 | Performance | High     | Tool bootstrap + manifest git storm on the GUI thread pre-paint; msleep(500)+wait(1000)            |
| 61  | #8939 | Performance | High     | Sidekick readiness: blocking HTTP on the GUI thread at 2 Hz for 45 s                               |
| 62  | #8940 | Performance | High     | Sync DB session + SELECT 1 resolved per request even with auth disabled                            |
| 63  | #8941 | Performance | High     | Four REST polling loops (one 5 Hz); analysis endpoints CPU-bound on the event loop                 |
| 64  | #8942 | Performance | High     | File-transport syscalls per message + 30 Hz polling; codemap blake3 retry + per-symbol resplit     |
| 65  | #8943 | Performance | Med-High | Batch: plot-data orchestrator per call, URDF re-parse per request, pandas at boot                  |

## Synthesis

**UI/UX.** The desktop surface has one systemic root cause (the broken
shared-theme import, #8894, verified by execution) plus a systemic
pattern: heavy work on the GUI thread with no worker/progress/cancel
(#8880, #8898, #8903, #8938, #8939). Destructive actions lack
confirmation across tools (#8881, #8884, #8889), and two tools ship
enabled controls that do nothing (#8879 crashes outright, #8882, #8883).
The good patterns exist in-repo (RunFitButton worker, terrain theme
helper, simulation-backends unavailable-state widget) — the fixes are
mostly promotion of existing patterns to shared helpers.

**Performance.** Three tiers: (1) cold start is dominated by real
imports of every physics runtime on both launcher and API start
(#8934) plus GUI-thread bootstrap (#8938); (2) the mocap/IK pipeline
wastes orders of magnitude via unvectorized Python loops and redundant
FK/mj_forward calls (#8921-#8927) — with already-built Rust kernels
sitting uncalled (#8927); (3) interactive surfaces stutter from
per-paint dynamics recomputation, per-request engine rebuilds, and
event-loop blocking (#8929, #8935, #8936, #8941). Top Rust candidates,
in order: whole-trajectory IK solve, normalization batch matmul,
wiring the existing gap-fill kernels, extending the ball-flight
integrator with turbulence.

**Proximal–distal modeling.** Documentation discipline is unusually
good, but two verification controls are void: the distributed-grip
atlas's cross-engine parity compared MuJoCo to itself (verified in the
committed artifact: `pinocchio: "0.1"`, exact-zero errors — #8909), and
the manufactured-solution controls are algebraic identities (#8910).
#8751/#8752 must not close on current evidence. The most
conclusion-threatening pair is #8913+#8914 (effective n≈24 with the
sign tally pooling nested time steps, and no comparison of effect size
to the numerical dissipation floor). Fix order: #8910, #8909, then
#8913/#8914, then bounding work (#8911, #8912, #8915-#8917).

## Status

Review complete: 65 issues filed (#8879–#8943) on 2026-08-21. Combined
with wave 1 (#8816–#8876), the two-day adversarial review filed 111
issues total, all with file:line evidence and fix instructions.

## Status

Review in progress. If usage runs out, the filed issues are the durable
output; the table above is the index.
