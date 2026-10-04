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
5. Call public `export_replay_impact_receipt(result.swing_state, trajectory_path,
receipt_path)` to publish a companion JSON receipt. Existing destinations are
   refused. Keep both files together: the receipt binds the exact trajectory bytes
   by SHA-256 and byte size and retains all state vectors, effective mass/inertia,
   optional impact offset and the full extraction metadata.
6. Recall with `load_replay_impact_receipt(receipt_path, trajectory_path)`.
   It verifies the exact trajectory and canonical viewer format before returning
   detached state arrays and metadata. It performs no SDK evaluation, impact
   simulation or flight resampling. It preserves the authored clock and false
   qualification flags. Recorded parent digests are portable provenance claims;
   loading a receipt does not freshly authenticate the original Library parents.
   The existing six-field trajectory wire and three-field aerodynamic provenance
   remain unchanged. The bounded application action below reauthenticates Library
   parents separately. Historical qualification remains acceptance work under #11464.

## Technical Methods Report

The supplemental [Authored Replay Impact Methods](necromatcher-replay-impact-methods.tex)
describes the extraction mathematics and software checkpoint `b059f04e`.
Its five-page PDF and editable source are saved on the local Desktop in
`Necromatcher Review 2026-10-01/Authored Replay Impact Methods Edition 1`.
All five pages were visually reviewed; the compile log has no layout warnings.
The built-in compiler failed to locate its runtime directories, so the existing
MiKTeX compiler produced the PDF with package installation disabled.
The [Report Review](historical_capture/replay-impact-methods-review-edition1.json)
pins the source, PDF, compile log and page renders. This edition predates the
portable receipt and React recall additions; the earlier Methods V8 is preserved.

## Application Replay Recall

In the Necromatcher page, choose a player and swing, then open Library Actions.
Select **Authored Replay HDF5** to import an existing authored replay using its
version ID and server-local source path. Import delegates to canonical replay
admission; it does not create rates or execute a replay from a kinematic fit.
The page labels registered replay assets explicitly rather than as controls.

Under **Authored Replay Recall**, choose a registered replay. The verified summary
shows its saved parents and hashes, sample count, backend, authored time step and
unqualified status. Missing or inconsistent qualification, sample, identity or
parent metadata blocks the summary and download link. Changing swings suppresses
late responses from the previous selection. **Download Verified Replay HDF5**
delegates to the existing backend revalidation route. This recall/import action
does not itself execute impact analysis; use the separate action below.

The focused React cohort passes 87 cases, including existing frame-domain
regressions. TypeScript and scoped ESLint pass.

## Application Impact Execution and Saved-Run Recall

After recalling a verified replay, use **Research Impact Preview**. Import a JSON
object with exactly `geometry` and `selection`, containing the existing typed
declaration records described above. Choose an existing replay sample and enter
an explicit execution budget in `(0, 600]` seconds. No contact selection, head
geometry, effective inertia or physical clock is supplied automatically.

**Preview Research Impact** dispatches an owned clean native worker through the
canonical matching service. The public workspace facade exports `NativeImpactSession`
and `impact_execution_stamp`. The worker reuses public replay extraction, the
existing trajectory coordinator, actual impact/flight pipeline and public receipt
export. The stamp extends refit provenance with physics/consumer sources and
installed flight-kernel file hashes. Stable source/runtime identities and canonical
replay parents are checked around calculation. Missing runtime capability fails;
production does not substitute a test flight.

The API exposes replay-owned routes:

- `POST /api/necromatcher/replays/{replay_id}/impact-runs`: explicit declarations and budget.
- `GET /api/necromatcher/replays/{replay_id}/impact-runs/{run_id}`: persisted, verified status.
- `POST /api/necromatcher/replays/{replay_id}/impact-runs/{run_id}/cancel`: owned cancellation.
- `GET /api/necromatcher/replays/{replay_id}/impact-runs/{run_id}/download`: authenticated bundle.

The client cannot select output paths. UUID run roots retain request and matching
manifests; complete output is staged exclusively before publication. Succeeded
research runs retain **rejected** scientific acceptance. Failed, cancelled,
incomplete, foreign or modified records cannot expose a verified download.
A restarted host can recall completed runs but cannot claim a live control handle
for an orphan running record.

Retain the displayed run ID. **Recall Research Impact Run** accepts that exact 32-hex ID
after reload and shows the result's saved geometry, selection, authored clock and
metrics. These declarations belong to the saved result, not a newly imported form
draft. Download contains `trajectory.json`, `impact-receipt.json`, `result.json` and
`request.json`. The result retains effective environment, impact/ball assumptions,
launch/post-impact state, metrics, execution identity and extraction metadata.
The current action uses pinned pipeline defaults; arbitrary environment or impact
model configuration is not exposed by this form.

Native PyQt tile parity is explicitly tracked in
[Issue #11487](https://github.com/D-sorganization/UpstreamDrift/issues/11487),
an epic #11232 child. React/Tauri uses the shared local API; the native tile's
replay import, impact controls and saved-run recall remain that follow-up's scope.

Extract `trajectory.json` and import it through the existing **Ball Flight** page
at `/ball-flight`. This imports retained samples without re-simulation. Keep the
receipt and result together when interpreting or sharing the curve; the six-field
trajectory alone does not retain full historical research provenance.

The [Application Methods Supplement](necromatcher-replay-impact-application-methods.tex)
documents admission, bounded publication, schemas and repeatability. Its four-page
PDF is compiled with the existing MiKTeX installation and installer disabled,
following an unavailable built-in compiler. All four pages are visually reviewed;
the final log has no layout warnings. Earlier methods editions remain unchanged.

Native handoff evidence uses a temporary canonical Library, synthetic model/profile,
independent MuJoCo replay, registered HDF5, the clean worker and actual Rust flight.
It verifies portable state, existing viewers, restarted recall and trajectory-byte
tamper rejection. No native/solver/flight/stamp/transport substitutions or historical
Library mutations occur in that opt-in acceptance. This is software integration
evidence, not a matched Tiger or Hogan swing. Tiger remains `[0,191)` and Hogan
`[0,750)`; the full goal remains active.

[Application Review](historical_capture/replay-impact-application-review.json) pins the reviewed implementation, native runtime build, test logs and Desktop bundle. The application checkpoint passes 125 focused Python cases with one Windows symlink privilege skip, one separate native handoff acceptance and 104 React cases. Configured mypy passes all six production owners.

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

The portable receipt continuation passes 69 combined Python cases, including the
earlier extraction/consumer checks and 31 receipt cases. Write/link/cleanup fault
tests preserve the primary failure and clean owned staging files after closing
their handles on Windows. Cold receipt loading passes with five native/GUI SDK
roots blocked. The public facade passes configured mypy and scoped Ruff/format.
The [Continuation Review](historical_capture/replay-impact-portability-and-recall-review.json)
pins both agent freezes, reviewed source/test bytes and final root test/type logs.

Both `physical_source_time_qualified` and `scientific_qualified` remain false.
These SI rates refer to authored simulation seconds, not calibrated Tiger/Hogan
motion time. Valid mathematical extraction does not qualify historical club
geometry, effective inertia, contact selection, impact predictions or golf play.
