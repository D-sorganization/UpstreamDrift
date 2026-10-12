# Necromatcher Explicit Coordinate Expansion

Tracked by #11281 under native fitting #11235 and epic #11232. This procedure changes the fitting coordinate hypothesis while preserving the parent polynomial motion before optimization. It does not establish historical wrist motion, anatomical geometry, depth or physical timing.

## Why the Wrist Trial Is Separate

Both rejected V10 player fits optimize 27 of 44 native coordinates. Four wrist coordinates (`LWInputX`, `LWInputY`, `RWInputX`, `RWInputY`) and two forearm coordinates (`LFInput`, `RFInput`) remain locked at zero. At the actual maximum-gap poses, public native grip Jacobians have rank four across the wrist columns, while derivatives of the current 13 projected landmarks with respect to these columns are zero. Adding forearm coordinates gives local grip rank six but changes image landmarks. These are properties of the declared generic model and attachment mapping, not observations of either player's wrists.

The current 27 free coordinates already provide local grip rank six. Closure is therefore not locally impossible, but satisfying it can move image-observed arms. Both V10 jobs exhausted their 30-evaluation budgets. Constraint-node gaps remain 36.779 mm Tiger and 10.042 mm Hogan, and dense independent gaps reach 68.451 mm and 15.368 mm. A coordinate experiment must report both fitted-node and independent interior results.

## Canonical Expansion Contract

Reuse the public `historical_fit.expand_image_spline_coordinates` and immutable `SplineCoordinateExpansion`. Supply the checked parent `ImageSplineStart`, an explicit immutable desired free-coordinate tuple and a complete finite reference pose in native coordinate order. Existing free names must remain present in their original relative order. Their reference positions must agree with the first knot within the declared absolute tolerance; they are validated, never overwritten.

The helper uses canonical Hermite `unpack` and `pack`. It copies every old position and velocity coefficient exactly and inserts constant reference positions with zero knot velocities for added coordinates. Native model identity, full coordinate order and knot times remain unchanged. The expanded free order, decision dimension and coefficient SHA-256 are explicit. The result retains the original coefficient hash, added coordinate names and detached reference pose.

This is not the parent's exact six-field restart: the selection and coefficient vector change. Canonical evaluation may differ by floating-point roundoff after the decision dimension changes. Independently verify complete native-order q/v/a equality at original source times, adjacent midpoints and knot/interior probes before optimization. Check that the parent's formerly locked coordinates are constant and equal the supplied reference throughout the original trajectory; the coefficient-expansion helper cannot establish that from a free-coordinate snapshot alone.

## Player Seed and Trial Procedure

1. Load the hash-verified stored fit and exact native binding through the existing library. Retain original model/XML/definition identities, source capture/frame identities, camera, attachment mapping, prior, observation weights and source clock.
2. Extract authored wrist limits from `NativeFitBinding.authored_coordinate_bounds()`. These remain generic authored limits, not historical measurements. Record all unbounded coordinates explicitly.
3. Expand only the four wrist coordinates first. Preserve the existing 27-coordinate coefficients and the full 44-coordinate initial trajectory. Do not change camera, geometry, contacts, training frames or solver budget in this comparison.
4. Validate the expanded snapshot through public `initialize_image_trajectory` with strict initialization and the complete original evidence/configuration plus the explicit expanded inputs. Reject infeasible seeds. Persist a separate unoptimized fit version with `optimizer_ran=false`, derived-seed provenance, original and expanded hashes and independently checked equality receipts.
5. Submit from that stored 31-coordinate seed through the canonical `NativeRefitSession`. A strict saved-spline restart of the new seed is permitted; describing it as an exact restart of the 27-coordinate V10 parent is not.
6. Freeze the implementation producer before actual jobs, bracket source/runtime and parent bytes, declare targets beforehand, and independently evaluate original training/held-out pixels, all scalar limits and sampled grip/rotation/penetration at original PTS, midpoints, phase boundaries and scalar extrema.
7. Save source-footage overlays, stills, receipts and reproduction scripts to a new Desktop review version. Update compact tracked evidence, development log, SPEC, turnover and the methods report. Successful execution and improved pixels do not imply physical qualification.

A later six-coordinate wrist-plus-forearm experiment must be a separately named hypothesis because it changes current image-landmark sensitivity. Camera fitting also requires an explicit gauge anchor; root motion and camera placement can compensate for one another. Retain the fixed camera in the first wrist experiment to isolate the coordinate effect.

## Native Saved Overlay Recall

Tracked by #11282 and #11246. Open the actual stored fit's **Export Fitted Overlay** dialog and choose its run from **Stored Overlay Exports**. The shared `NativeVideoSession.stored_runs(fit_id)` lists persisted statuses without scheduling work; `view_for_fit(fit_id, run_id)` verifies source-fit ownership and current parent identities/hashes before canonical status recall. A foreign fit, altered parent or malformed binding is rejected.

Select **Save Checked ZIP** only when the canonical artifact state permits it. The existing guarded download verifies source/video/still identities and archive bytes; the native transfer writes exclusively outside the library and checks destination bytes. Saving may create a new owned ZIP and refresh its verification baseline, but does not alter original fit or capture evidence. Scientific rejection and producer identity remain visible after saving. Offscreen QTest evidence verifies widget interactions; it is distinct from a manual native OS acceptance check.

## Matched Continuation Control

Compare each wrist trial with a separately stored 27-coordinate continuation from the same V10 parent, using the same first-pose prior, original observations, camera/contact configuration, 30-evaluation budget and 300-second guard. Both starts preserve the same complete native q/v/a motion within the declared tolerance. Earlier V10 optimization used an earlier parent prior and a separate optimization history; progress against V10 alone cannot isolate the effect of wrist freedom. Retain both the matched-control comparison and the historical V10 baseline, with separate immutable fit/run identities and independent assessments.
