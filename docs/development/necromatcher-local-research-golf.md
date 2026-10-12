# Necromatcher Local Research Golf

Issue #11493 connects a verified saved replay-impact result to the existing local
golf session. The input is a model-contact research hypothesis with assumed
contact, authored seconds and unverified scientific status. Successful delivery
to the local simulator does not qualify the historical model.

## Public Ownership and Admission

Use `workspace.load_research_impact_shot(library, replay_id, run_id, metadata)`.
The dedicated public impact session authenticates the saved parents, successful
rejected research manifest and exact four-file bundle. The companion receipt
checks retained extraction state against trajectory bytes. Saved result/request
parents, geometry, selection and execution identity must agree. Altered or
incomplete evidence is refused before a golf shot is prepared.

The public bridge uses the complete saved post-impact ball velocity and angular
velocity, not speed inferred from plotted positions. `ShotMetadata` supplies
explicit shot/session IDs, timestamp and a proper source-to-target aim rotation.
Model-run, trace digest, impact identity and authored sample time derive from
the authenticated result. Contact, numerical and scientific qualification stay
unverified. The returned immutable JSON context carries a detached whitelist of
parent hashes, authored-clock limitations and actual saved assumptions/settings.

## Frames and New Simulation

The source impact uses `flight_xfwd_yleft_zup`. A proper declared aim rotation
maps saved velocity and spin into the target local frame through the existing
spatial transformation owner. Preserve the saved receipt and original trajectory
unchanged; target-frame launch data is a separate declared transformation.

The Local Reference Simulator computes a **new flight** from that launch state
using its own model/settings and environment. Its output is not the original
retained impact trajectory and does not automatically adopt the original saved
environment. Both are research outputs. The local adapter declares no native
avatar, course feedback or aim-control support. A declared frame transform is
not a camera fit, historical contact event or physical-clock calibration.

## Explicit Operator Workflow

1. Recall a registered authored replay and a verified completed research impact
   run in Necromatcher. Choose **Open Local Research Simulation**.
2. Review the saved IDs, hashes, assumptions and authored sample time. The web
   route includes replay and run IDs; malformed research requests do not fall
   back to the manual demo.
3. Choose **Connect**, then **Prepare Research Impact** on the web or
   **Prepare Research Shot** on desktop. Preparation authenticates
   stored evidence and does not arm or submit the shot.
4. Choose **Arm**, then explicitly **Submit**. Disarm or cancel pending shots
   through their actual owned handles. A stale request or changed source/session
   must not retain an old preparation or token.
5. Inspect the local delivery receipt and retained samples. Delivery acceptance
   describes execution in the local simulator, independently of unverified
   scientific acceptance. Web positions use the existing flight visualization;
   native retained values remain available for inspection.

`GolfSessionService.prepare_research_shot` requires the actual connected local
adapter, supported shot input and local trajectory return, explicit model
provenance and all three unverified qualification axes. Ordinary `prepare`
retains its qualified-contact requirement. Research cannot use fake, relay or
external destinations. Destination changes invalidate pending preparation, and
the restriction is checked again before submission. Local flight runs outside
the event loop through the existing simulation owner.

## API and Recall Limits

After connecting a matching local session, post explicit IDs, timestamp, aim and
context revision to `/tools/golf-simulator/shot/prepare-research-impact`. Normal
arm/disarm/cancel/submit endpoints retain their existing lifecycle. Read
`/tools/golf-simulator/shot/{shot_id}/local-trajectory` only after confirmed owned
local delivery. The response retains the source research context alongside the
new local samples and simulator provenance. Reused shot IDs cannot replace a
prior source context. Foreign session responses and incomplete runs fail closed.

Registered replays and original impact results persist in the canonical Library.
The current local golf session and its trajectory/context are session data;
they are not claimed as restart-persistent golf simulations. After restarting,
recall the saved replay/impact run and explicitly create a new local simulation.

## Verification and Historical Limits

Boundary tests must cover qualification promotion, full asymmetric vectors,
proper nonidentity aim, foreign/stale/tampered bundles, destination replacement,
single-use tokens, cancellation and detached retained samples. The actual native
acceptance must use a fresh temporary canonical Library, real MuJoCo replay,
clean-worker impact, Rust flight and real local preparation/submission. Ordinary
service or GUI doubles establish boundary behavior only.

The independent Python cohort passes 209 cases with no skips; an additional
actual API/Qt acceptance passes with real MuJoCo replay, a clean impact worker,
Rust flight and seven retained local samples. Native recall reproduces the
entire table without recomputing the flight. The React cohort passes 106 cases;
configured mypy passes on seven production owners. See the
[Frozen Review](historical_capture/local-research-golf-review.json) for exact
source hashes, logs, synthetic screenshots and report artifacts. These counts
describe this checkpoint and do not establish historical fit qualification.

Tiger stays within original frames 0–190, `[0,191)`; Hogan stays `[0,750)`.
Camera, anatomy, continuous ROM/contact, physical source time and shaft-distortion
qualification remain active. Local research golf does not establish a qualified
historical swing, native avatar animation, course integration, external delivery
or engineering release approval.

## Local Research Admission Race Follow-Up — #11493

A concurrent request regression reproduced reused shot-ID admission after another
request prepared and cancelled during authentication. The API now applies the
same identity guard before and immediately after authentication. All 10 API
cases pass; a fresh additional actual MuJoCo/clean-worker/Rust API and Qt
acceptance passes in 24.56 s, with seven exact retained samples and no
re-simulation on recall. Configured mypy passes for the changed API owner.
The earlier 209-case cohort and 106 React cases remain separate checkpoint
evidence. The follow-up preserves unverified qualification and all historical
limits; the full goal remains active.

See [Follow-Up Review](historical_capture/local-research-golf-admission-race-review.json). Source inventory covers 6,697 src/test
files, with equal before/after maps; this is not a whole-repository baseline.
