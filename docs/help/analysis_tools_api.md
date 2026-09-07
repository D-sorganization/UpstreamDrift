---
title: Analysis Tools
tile_id: analysis_tools_api
status: complete
---

# Analysis Tools

## Purpose

Analysis Tools is the REST surface over a running simulation: it samples
metrics from the active physics engine, keeps a bounded history, summarises
that history statistically, exports it, and offers two body-space measurement
helpers. The tile is web-only (`surfaces: ["web"]`, web page
`/tools/analysis`); its handlers live in `src/api/routes/analysis_tools.py`.

This page is the single in-GUI calculation reference for the analysis stack.
Every formula below is the formula the code actually evaluates, with the module
that evaluates it named. Where a quantity a reader might expect is not
computed, that is stated in [Limitations](#limitations) rather than papered
over.

## Inputs

The endpoints take almost no numeric input: the state comes from the active
engine, not from the caller.

| Input | Source | Unit |
| --- | --- | --- |
| Generalized positions $q$ | `engine.get_state()` | rad (revolute) or m (prismatic) |
| Generalized velocities $v$ | `engine.get_state()` | rad/s or m/s |
| Mass matrix $M(q)$ | `engine.compute_mass_matrix()` | mixed, per degree-of-freedom pair |
| Body Jacobian $J$ | `engine.compute_jacobian(body)` | m per rad (linear rows) |
| Simulation time | `engine.time` | s |
| Applied joint torques | `engine.get_applied_torques()` (optional) | N m |
| `format` | `POST /analysis/export` body | `"csv"` or `"json"` |
| `time_range` | `POST /analysis/export` body, `[start, end]` | s |
| `body_name`, `position`, `rotation` | `POST /simulation/position` body | name, m, rad (roll, pitch, yaw) |
| `body_a`, `body_b` | `POST /simulation/measure` body | body names |
| `drift_generalized_force`, `control_generalized_force` | `POST /analysis/tools/drift-control/ratio` | N or N m per degree of freedom |
| `decay_rate`, `horizon`, `n_steps`, `n_trials`, `perturbation_scale` | `POST /analysis/tools/contraction/estimate` | 1/s, s, count, count, dimensionless |

## Outputs

| Output | Endpoint | Unit |
| --- | --- | --- |
| `sim_time` | `/analysis/metrics`, `/analysis/statistics` | s |
| `joint_positions`, `joint_velocities` | `/analysis/metrics` | rad or m; rad/s or m/s |
| `max_velocity` | `/analysis/metrics` | rad/s or m/s |
| `rms_velocity` | `/analysis/metrics` | rad/s or m/s |
| `kinetic_energy` | `/analysis/metrics` | J |
| `club_head_speed` | `/analysis/metrics` | m/s |
| `current`, `minimum`, `maximum`, `mean`, `std_dev` per metric | `/analysis/statistics` | same unit as that metric |
| `sample_count` | `/analysis/statistics` | count (at most 500) |
| `time_series` (metric name to values) | `/analysis/statistics` | same unit as that metric |
| CSV or JSON attachment | `/analysis/export` | file download |
| `distance`, `position_a`, `position_b`, `delta` | `/simulation/measure` | m |
| `angle_rad`, `angle_deg`, `velocity`, `torque` per joint | `/simulation/measurements` | rad, deg, rad/s, N m |
| `ratio`, `summary` | `/analysis/tools/drift-control/ratio` | dimensionless |
| `estimated_rate`, `is_contracting` | `/analysis/tools/contraction/estimate` | 1/s, boolean |

Endpoint paths are as declared in the router. `register_routes` in
`src/api/route_registry.py` prepends a deployment-configured prefix; the tile
declares `/api/analysis`.

## Method

### Energy Analysis
**Kinetic energy.** The API computes, per sample,

$$T = \tfrac{1}{2}\, v^{\mathsf T} M(q)\, v \quad [\mathrm{J}]$$

in `_collect_metrics` (`src/api/routes/analysis_tools.py`), evaluated as
`0.5 * v @ M @ v`. It is skipped when `compute_mass_matrix()` returns `None`.

**Potential energy and total energy.** These are not produced by the API layer.
They come from the engine recorders. The Pinocchio backends
(`src/engines/physics_engines/pinocchio/python/pinocchio_golf/gui_simulation.py`
and the sibling `motion_matching/simulate.py`) record

$$T = \texttt{pin.computeKineticEnergy}(model, data, q, v), \quad
V = \texttt{pin.computePotentialEnergy}(model, data, q), \quad E = T + V$$

Repo gotcha: `pinocchio.computeTotalEnergy` does not exist in the Python
bindings this repo depends on. Every call site calls `computeKineticEnergy` and
`computePotentialEnergy` separately and sums them. The rule is recorded in
`src/engines/physics_engines/pinocchio/PINOCCHIO_PARITY_SPEC.md` (section 2.2)
and echoed in the docstrings of the motion-matching modules. Do not collapse a
call site to `computeTotalEnergy`.

**Energy summary metrics.** `EnergyMetricsMixin.compute_energy_metrics`
(`src/shared/python/analysis/energy_metrics.py`) takes the kinetic and
potential time series and returns, with $E_k = T_k + V_k$:

| Metric | Formula | Unit |
| --- | --- | --- |
| `max_kinetic_energy` | $\max_k T_k$ | J |
| `max_potential_energy` | $\max_k V_k$ | J |
| `max_total_energy` | $\max_k E_k$ | J |
| `energy_efficiency` | $100 \cdot T_{k^*} / \max_k E_k$, where $k^*$ is the index of peak club head speed | percent |
| `energy_variation` | standard deviation of $E_k$ | J |
| `energy_drift` | $E_{N-1} - E_0$ | J |

`energy_efficiency` is `0.0` when no club-head-speed series is attached to the
instance, and `0.0` when the maximum total energy is not positive.
Preconditions require equal-length finite arrays with $T_k \ge 0$.

**Energy-conservation checking.** `check_energy_conservation`
(`src/shared/python/simulation_backends/validation.py`) rolls the backend
forward passively (zero torque) for `horizon` steps of `dt`, evaluates
$T_k = \tfrac{1}{2} v_k^{\mathsf T} M(q_k) v_k$ at every step, and reports

$$\varepsilon = \max_k \frac{|T_k - T_0|}{|T_0|}, \qquad
\text{passed} \iff \varepsilon \le \text{rel-tol} \; (\text{default } 10^{-2})$$

The docstring is explicit that this is valid only for a model configured
conservative (gravity disabled, zero joint damping), because kinetic energy is
then the only energy store along a passive rollout. It raises when $T_0 = 0$,
since a relative tolerance has no scale.

**Energy transfer.** What the repo implements is a whole-system energy rate,
not a segment-to-segment transfer. `plot_energy_flow`
(`src/shared/python/plotting/energy.py`) computes

$$\dot E_k = \frac{E_k - E_{k-1}}{t_k - t_{k-1} + 10^{-10}} \quad [\mathrm{W}]$$

from the recorded kinetic and potential series, shading a non-negative rate as
Energy Input and a negative rate as Energy Output. There is no per-segment
energy accounting in the analysis package; inter-segmental power flow is listed
as a planned enhancement in the docstring of
`src/shared/python/biomechanics/kinematic_sequence.py`.

### Kinematic Sequence
`SegmentTimingAnalyzer.analyze(segment_velocities, times)`
(`src/shared/python/biomechanics/kinematic_sequence.py`;
`KinematicSequenceAnalyzer` and `KinematicSequenceResult` are
backward-compatible aliases) takes a mapping of segment name to 1-D velocity
array plus a time array.

| Quantity | Formula | Unit |
| --- | --- | --- |
| `peak_velocity` | maximum absolute velocity, per segment | unit of the supplied velocity (rad/s for recorder joint velocities) |
| `time`, `index` | time and index at that maximum | s, index |
| `normalized_velocity` | segment peak divided by the largest peak in the set | 0 to 1 |
| `speed_gain` | this segment's peak divided by the previous entry's peak in the caller's `expected_order`, only when the proximal peak exceeds $10^{-6}$ | dimensionless |
| `deceleration_rate` | negative slope of the absolute velocity across a 30 ms window starting at the peak | velocity unit per s |
| `timing_gaps` | difference of consecutive peak times over peaks sorted by time, keyed `"a->b"` | s |
| `sequence_consistency` | correctly ordered pairs divided by comparable pairs, over all two-combinations of the `expected_order` segments present in the data | 0 to 1 |
| `is_valid_sequence` | `sequence_consistency == 1.0` and at least two peaks | boolean |

Proximal-to-distal sequencing is not built in. `expected_order` is a
caller-supplied list. The module docstring states plainly that it implements no
proprietary methodology and treats the order as a neutral, user-defined
parameter. With no `expected_order`, `sequence_consistency` is `0.0`,
`is_valid_sequence` is `False`, and no `speed_gain` is computed. Consistency is
scored pairwise rather than by absolute index, so one out-of-order segment
degrades the score instead of zeroing it. Edge cases: exactly one peak yields
consistency `1.0`; more than one peak with no comparable pair yields `0.0`. The
result carries `methodology = CITATION_SEGMENT_TIMING` (see
[Method citations](#method-citations)).

**X-Factor.** Two different X-factor computations exist, and they are not
interchangeable.

1. `SwingMetricsMixin` (`src/shared/python/analysis/swing_metrics.py`):

   $$X(t) = \theta_{\text{shoulder}}(t) - \theta_{\text{hip}}(t) \quad [\mathrm{deg}]$$

   from two caller-chosen columns of `joint_positions`, converted with
   `np.rad2deg`. `compute_x_factor_stretch` returns the finite-difference
   derivative `np.gradient(X, dt)` in deg/s together with the peak stretch
   rate, the maximum of its absolute value.

2. `XFactorMetrics` in `src/shared/python/injury/spinal_load_analysis.py`
   carries the pelvis-thorax separation series (deg), `x_factor_stretch` as the
   maximum separation angle (deg), its time (s), `separation_rate` (deg/s) and
   `transition_duration` (s). `SpinalLoadAnalyzer` grades it against the
   thresholds declared as class constants: safe below 45 deg, caution 45 to 55
   deg, high risk above 55 deg, with 65 deg as the `X_FACTOR_HIGH` bound.

Neither variant is exposed by the analysis endpoints; both are library calls.

### Jacobian Analysis
**End-effector velocity mapping.** For a body Jacobian $J(q)$,

$$\dot x = J(q)\, v$$

The API uses this directly for club head speed in `_collect_metrics`: it takes
the `"linear"` block of `engine.compute_jacobian("club_head")` and reports the
Euclidean norm of the resulting velocity in m/s, evaluated with `math.hypot`
over the components.

**Manipulability ellipsoid.** `compute_manipulability_ellipsoid`
(`src/shared/python/spatial_algebra/manipulability.py`) takes the singular
value decomposition $J = U \Sigma V^{\mathsf T}$ (`full_matrices=False`) and
returns the singular values as the ellipsoid principal radii and the right
singular vectors as its principal axes. Long axes are the easily moved
directions; a collapsed axis is a singular configuration. The module notes the
dual reading: velocity manipulability from $J \dot q = \dot x$, force
manipulability from $J^{\mathsf T} f = \tau$.

**Yoshikawa manipulability index.**

$$\mu = \sqrt{\det\left(J J^{\mathsf T}\right)} = \prod_i \sigma_i \quad [\text{dimensionless}]$$

implemented as `np.prod(np.linalg.svd(J, compute_uv=False))`. A value of zero
is a singular configuration (a lost degree of freedom); larger is further from
singularity. Reference given in the source: Yoshikawa, T. (1985),
"Manipulability of Robotic Mechanisms", The International Journal of Robotics
Research 4(2), 3-9.

**Conditioning and singularity thresholds.** `check_jacobian_conditioning` in
the same module computes $\kappa = \sigma_{\max} / \sigma_{\min}$ via
`np.linalg.cond` and applies the Guideline C2 thresholds:

| Condition | Behaviour |
| --- | --- |
| $\kappa > 10^{6}$ | warning logged: near-singularity, reduced manipulability |
| $\kappa > 10^{10}$ | error logged; switch to pseudoinverse or damped least squares |
| $\kappa > 10^{12}$ | raises `SingularityError`; the configuration is treated as unrecoverable |

An empty Jacobian returns infinity rather than raising.
`src/engines/common/jacobian_diagnostics.py` wraps the same decomposition into
a `JacobianDiagnostics` record that adds the numerical `rank` (singular values
above `RANK_TOLERANCE = 1e-8`), `nullspace_dim = n - rank`, the full
singular-value spectrum and `is_near_singular` at the first threshold, over the
golf task points `clubhead`, `grip`, `left_hand`, `right_hand` and
`shaft_mid`.

### Statistics, Export and Measurement
`/analysis/metrics` appends each snapshot to a bounded in-memory buffer
(`max_history = 500`, oldest dropped). `/analysis/statistics` scans that buffer
for scalar, non-NaN keys and reports per key the minimum, maximum, mean and
standard deviation via NumPy (population standard deviation, `np.std` default
`ddof=0`) plus the last value as `current`. `/analysis/export` streams the same
buffer as CSV (numeric columns only, sorted field names) or JSON, optionally
filtered to a `sim_time` range.

`/simulation/measure` returns the Euclidean distance `math.dist(pos_a, pos_b)`
in metres and the componentwise delta from body A to body B.
`/simulation/measurements` reports every generalized coordinate with
`angle_rad`, `angle_deg = math.degrees(angle_rad)`, its velocity and its
applied torque, falling back to `joint_<i>` names and `0.0` torques when the
engine does not expose them.

`/analysis/tools/drift-control/ratio` computes the realized drift-to-input
ratio, the norm of the drift generalized force divided by the norm of the
control generalized force, from caller-supplied arrays
(`src/tools/drift_control/analyzer.py`).
`/analysis/tools/contraction/estimate` estimates a local contraction rate from
nearby deterministic rollouts (`src/tools/contraction/verifier.py`).

## Method Citations
`src/shared/python/analysis/dataclasses.py` defines a frozen `MethodCitation`
dataclass (`name`, `authors`, `year`, `title`, `doi`, `notes`) and four
pre-defined citations. They are attached to analysis results as a `methodology`
field. This table reproduces what that module declares.

| Name | Authors | Year | Title | DOI |
| --- | --- | --- | --- | --- |
| Proximal-to-Distal Sequencing | Putnam | 1993 | Sequential motions of body segments in striking and throwing skills | `10.1016/0021-9290(93)90084-R` |
| X-Factor | Cheetham et al. | 2001 | The importance of stretching the X-Factor in the downswing of golf | none declared in the source |
| Crunch Factor | McHardy and Pollard | 2005 | Muscle activity during the golf swing | `10.1136/bjsm.2004.014514` |
| Spinal Load Analysis | Hosea et al. | 1990 | Biomechanical analysis of the golfer's back | none declared in the source |

The `notes` field on each citation qualifies the method:

- Proximal-to-Distal Sequencing (`CITATION_SEGMENT_TIMING`, aliased
  `CITATION_KINEMATIC_SEQUENCE`): "User-supplied expected order; no
  proprietary methodology." Attached to `SegmentTimingResult.methodology`.
- X-Factor (`CITATION_X_FACTOR`): "Pelvis-thorax separation angle and
  stretch-shortening cycle." Attached to `XFactorMetrics.methodology`.
- Crunch Factor (`CITATION_CRUNCH_FACTOR`): "Lateral bend + axial rotation
  coupling metric." Attached to `CrunchFactorMetrics.methodology`.
- Spinal Load Analysis (`CITATION_SPINAL_LOAD`): "Up to 8x body weight
  compression; validated by Lindsay et al. (2002)." Attached to
  `SpinalLoadResult.methodology`.

Two of the four citations carry no DOI in the source. That is a gap in the
citation records, not an omission on this page: the record the code ships is
what a report or audit log will carry, so no substitute identifier is supplied
here.

## Limitations

- No potential energy at the API layer. `/analysis/metrics` computes kinetic
  energy only. It therefore cannot report total energy, cannot check energy
  conservation, and its `kinetic_energy` series alone is not a conservation
  diagnostic for a model with gravity enabled.
- No inter-segment energy transfer. The only flow quantity is the whole-system
  rate of change of energy in `plot_energy_flow`. Segment-level energy
  accounting and inter-segmental power flow are not implemented.
- `check_energy_conservation` is a two-degree-of-freedom helper. Its internals
  validate a length-2 position, a length-2 velocity and a 2-by-2 mass matrix.
  It is not a general whole-model conservation check, and it is meaningful only
  on a backend explicitly configured conservative.
- No proximal-to-distal order is asserted. The segment-timing analyzer scores
  whatever order the caller supplies. It does not know that the pelvis precedes
  the torso, and it publishes no reference peak-velocity ranges.
- `/simulation/measurements` assumes revolute joints. It calls `math.degrees`
  on every generalized coordinate, so a prismatic degree of freedom is reported
  in meaningless degrees. Torques default to `0.0` when the engine has no
  `get_applied_torques`.
- Metric history is volatile and bounded. Five hundred snapshots are held in
  memory on the engine manager with no persistence. A restart loses them and
  `/analysis/export` returns HTTP 400 until a simulation has run again.
- Statistics are per-sample, not per-unit-time. Snapshots are taken whenever
  `/analysis/metrics` is polled, so the mean and standard deviation are sample
  statistics rather than time averages; an irregular polling cadence biases
  them.
- No engine capability negotiation. A missing engine feature is handled by
  omitting a metric (energy, club head speed) or by returning HTTP 501, not by
  a declared capability set.
- Nothing on this tile validates a model across engines.
  `validate_timing_cross_engine` (5 ms tolerance) and
  `validate_angle_cross_engine` (2 deg tolerance) in
  `src/shared/python/analysis/dataclasses.py` compare two engines and are not
  called by any endpoint here.

## See Also
- [Analysis Tools narrative guide](analysis_tools.md)
- [Simulation Controls](simulation_controls.md)
- [Engine Selection](engine_selection.md)
- [Visualization](visualization.md)
- [Project Map](../architecture/PROJECT_MAP.md)
- [User manual](../user_guide/user_manual.md)
