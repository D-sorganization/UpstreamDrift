# Necromatcher Authored Replay Procedure

## Execute an Independent Run

Use `workspace.ReplayOptions(source_frame_index, initial_rates, duration_s, dt_s)`
and `workspace.replay_authored_profile(library, profile_id, options)` in the
existing clean-interpreter SDK worker context. Supply all native coordinate
rates explicitly in ordered m/s and rad/s; they are operator assumptions.
Choose a duration containing an integer number of steps within the profile
horizon. Default refinement is four; the fine run is limited to 4096 steps.
The selected source frame initializes the pose and does not infer alignment
between footage PTS and authored seconds.

The implementation reuses FullBodySimulator RK4, native mass/gravity, contact
and closure dynamics. Controls depend on authored time only. Saved initial
poses and supplied rates are preserved. Entire root polynomials must be zero,
including interior coefficients. Unsupported native capabilities fail closed.

## Inspect the Result

The canonical simulation Trace records commands evaluated at saved state times,
rather than the simulator record's preceding-step effort. Ordered m/rad and
N/N\*m units, identities, source frame and initial rates use JSON strings so
canonical HDF5 metadata preserves them. Mixed generalized efforts are not
labelled torques. Use `simulation_backends.trace_io.write_trace/read_trace` for
local round trips. Immutable library registration and unit-aware downstream
analysis admission remain to be implemented.

A fresh compiled resource repeats integration at the finer step. Translation
and rotation differences are reported independently. Initial and maximum grip
gaps remain geometric diagnostics. Scientific and physical source-time
qualification remain false regardless of numerical agreement.

## Validation and Limits

Ten red-first tests exercise real native integration, fresh-resource refinement,
nonzero joint commands, exact initial state, authored clock and canonical HDF5
recall; they reject invalid options, nonzero interior root commands, missing
frames, wrong rate counts and excessive horizons. The original all-zero
synthetic pose penetrated the ground and failed on nonfinite dynamics. The
positive fixture explicitly lifts its synthetic pose clear of the ground;
production replay never changes saved initial geometry.

Actual historical fits have large grip gaps, unknown source clocks and generic
anatomy. The bridge does not establish accepted historical motion, measured
controls or a qualified golf shot. Continue initial-state admissibility,
immutable replay storage and impact/whole-analysis consumers under #11235 and
#11232. PR #11240 remains draft; the documented CI remediation budget remains
exhausted.
