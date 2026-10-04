# Necromatcher Authored Replay Impact Extraction

## Boundary and Scientific Status

Issue #11464 progresses epic #11232. The public workspace facade exports
`ReplayImpactGeometry`, `ReplayImpactSelection` and `extract_replay_impact_state`.
Extraction reads an authenticated stored authored replay, preserving exact replay,
profile, fit, model and capture identities and hashes in `SwingState.metadata`.
It never substitutes a reference swing or differentiates video presentation time.

The operator declares a body-local clubhead point, unit perpendicular face normal
and face-up vectors, effective impact mass and scalar moment of inertia. These are
explicit assumptions, not inferred historical measurements. The selected recorded
sample and proper rigid world-to-flight transform are also authored. A selected
sample is not an automatically detected collision.

## Native Kinematics and Units

The existing public native marker linearizer evaluates the head point and three
mathematical face-basis probes. Probe length is explicitly 1 m; the probes are
derivative constructions, not a one-metre physical clubhead.

For recorded generalized rates $v$ on authored simulation seconds, point velocity
is $\dot p=J(q)v$. Mixed translational and rotational coordinates retain their
compiled metre/radian order and corresponding rate units. Three orthonormal world
axes are obtained as $e_i=(p_i-p_0)/L$ and their derivatives as
$\dot e_i=(\dot p_i-\dot p_0)/L$. Rigid-body angular velocity is
$\omega=\frac12\sum_{i=1}^3 e_i\times\dot e_i$.

The extraction checks probe identity/order, public body-pose agreement, proper
rotation and preservation of the orthonormal triad by its derivatives. Missing
derivative capability fails explicitly; angular velocity is not filled with zeros.
The declared flight rotation maps all vectors. Translation affects only the
reported head-point position. Metadata retains the recorded replay sample/time,
initial capture frame, assumptions and `capture_pts_used_for_velocity=false`.

## Repeatable Consumer Procedure

1. Recall an independently stored authored replay using its immutable ID. Establish
   the normal SDK-first native context for the selected engine.
2. Author and review the typed geometry and sample/flight transform. Reuse exact
   native model frames; distinguish body origins, named frames and solid centres.
3. Call the public `extract_replay_impact_state(library, replay_id, geometry,
selection)`. Canonical authenticated reads recheck asset bytes and parents.
4. Pass the returned state to existing
   `ShotTrajectoryHandoffCoordinator.simulate_and_export_trajectory` with a reviewed
   output name. Declared `FLIGHT_FRAME_ID` states retain full velocity, angular
   velocity and face normal. An incompatible declared frame is rejected before
   output. Legacy undeclared provider conversion retains its existing behavior.
5. Retain the returned `PipelineResult` and its metadata with the trajectory.
   The existing six-field trajectory wire carries aerodynamic provenance and
   retained flight samples; it does not yet persist the full replay extraction
   receipt. Portable receipt storage and a normal app action remain acceptance
   work under #11464. Do not close the issue on extraction alone.

## Verification and Limits

Meaningful RED cases reproduced full-vector loss and silent frame conversion.
Independent synthetic screw motion checks translational/rotational velocity,
flight rotation, explicit assumptions, stale parents, order/units, malformed
metadata, unsupported derivatives, nonrigid probes and mid-read parent changes.
The public facade is cold-importable with five engine/GUI SDK imports blocked.

Root passed 38 cases combining extraction with the actual impact solver and
existing trajectory-viewer importers. The integration fixture substitutes flight
integration when its optional Rust wheel is absent; it does not substitute the
impact solver. Configured mypy, Ruff and formatting pass. No historical native
replay or Library mutation occurred during this software checkpoint.

Both `physical_source_time_qualified` and `scientific_qualified` remain false.
These SI rates refer to authored simulation seconds, not calibrated Tiger/Hogan
motion time. Valid mathematical extraction does not qualify historical club
geometry, effective inertia, contact selection, impact predictions or golf play.
