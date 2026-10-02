# Necromatcher Authored Contact Phases

Tracked by #11235 under #11232, Tiger #11226 and Hogan #11229. Contact phases are source-clock research hypotheses. They are not measured contact forces, physical swing timing, no-slip constraints or historical anatomy.

## Source Review Before Authoring

The Desktop `Contact Phase Review V10` package contains 32 unchanged original PNGs, detached detector observations, exact rational PTS, decoded-frame identities and PNG-byte hashes. Its full receipt SHA-256 is `88973ff55f29df194d8e1e6b133558dab74a743891ef922e3c5fadbfea4889cd`. Preserve the original captures and full intervals; a reviewed subinterval is an additional hypothesis.

Tiger's trail heel is apparently low at frame 80 (17.684333 source seconds), transitional/blurred at 85, and visibly raised with a toe-pivot silhouette at frame 90 (18.018). Thus frames 80–90 bracket a release uncertainty; they do not identify a measured release instant. Both shoes remain inspectable at 105 (18.5185), while cropping begins by 110 (18.685333). Frames 0–105 are a conservative review interval including early follow-through. Completion of the finish and later walking cannot be inferred from cropped feet.

Hogan frames 0–499 (110–126.633333 source seconds) retain the body and shoes during the first swing and early follow-through. At 320×240 resolution, blur prevents a defensible heel-release boundary. Frame 499 does not certify a completed finish. Detector foot/heel confidence is inferred visibility and cannot resolve the ground gap or repair clipping.

An unknown phase uses an empty equality-pin set. A plausible visible stance may support an explicitly authored pin hypothesis, with the relevant source review hashes retained. Report alternative boundary choices and their image/constraint effects rather than presenting one operator choice as truth.

## Canonical Contract

`historical_fit.ContactPinPhase` retains exact rational `start_pts` and `end_pts`, immutable named `pinned_spheres`, and immutable `review_frame_sha256` references. `ContactPinSchedule` binds ordered phases to the immutable capture ID/hash and requires `status=authored_contact_hypothesis`. `ScheduledConstraintOptions` wraps the existing native `ConstraintOptions` in `base` and `schedule`; `ImageFitConfig` keeps its existing eight fields.

Phases cover the complete fitted source interval without gaps or overlaps. Boundaries are half-open; the final endpoint is included in the final phase. Numeric boundaries must remain distinct after conversion to the solver's source-time floats. Resolve each evaluation time to the existing native options using the active pin set. Include phase boundaries in constraint assessment while retaining the original image observations and their weights.

Pinned sphere rows enforce signed normal height relative to the declared plane. Released rows retain unilateral nonpenetration. Residual row labels and dimensions remain stable across phases. This does not anchor the sphere tangentially, infer friction or enforce a toe pivot. The base ground/grip scales and weights remain authored model assumptions.

## Capture Binding and Persistence

The shared workspace contact binder runs before native optimization. It checks the schedule capture ID/hash against the hash-verified parent and open capture, verifies source asset/shot/camera identity, resolves every boundary to an exact original frame PTS, and resolves each reviewed decoded-frame hash to one frame within its phase. A hash belonging to another phase or ambiguous repeated image is rejected. Capture frame hashes use bare hexadecimal in their existing schema; schedule references use the canonical `sha256:` prefix. PNG-byte hashes identify a different byte representation and must not be substituted for decoded-frame hashes.

Keep the complete nested configuration in saved request options and original-fit evidence. Native/web recipe recall must disclose the authored phase count and interval and retain the wrapper unchanged when editing evaluation budgets or scalar weights. The native worker independently validates binding; a successful numerical job remains rejected until independent scientific acceptance is established.

The exact saved-spline start still requires the original full knot interval and coordinate identities. Trimming the fitted interval requires an explicitly different initialization path; never describe resampling a shorter interval as an exact coefficient restart.

## Validation and Next Trials

Meaningful checks cover legacy equivalence, exact phase boundary semantics, malformed identity/hash/clock records, released positive-height residuals with continuing penetration penalties, stable labels, native Jacobian finite differences, serialization, unchanged image-observation count/RMS, and SDK-free capture binding failures. Reuse the existing canonical fit/job/native providers.

Before historical phase trials, freeze the implementation producer, retain the reviewed source hashes, declare image and contact tolerances, and save the actual authored schedule. Assess original training/held-out image residuals, hard scalar bounds, grip closure and ground penetration at source PTS, phase boundaries, interior probes and scalar extrema. Finite nonlinear sampling remains `continuous_certified=false`. Actual rejected V10 trials and independent target failures are retained in [V10 Summary](historical_capture/phase-contact-v10-summary.json); source review remains separate from optimization and physical contact acceptance.
