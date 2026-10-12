# Arm Morphology Admission and Continuation

## Accepted Checkpoint

Four immutable, unoptimized arm-length hypotheses were admitted from Tiger variant V20 and Hogan variant V17 using published producer `1f7fe89be67d98ca9db307905a08aa54c2d53fb2`. The public native hypothesis pipeline authenticated source, model, camera, marker mappings and saved-final motion before publication. No optimizer ran and no player anatomy was accepted. Documentation commit: SELF. Tracking: #11357, #11394 and continuation PR #11359.

[Compact Results and Exact Pins](historical_capture/arm-morphology-admission-summary.json) records the four model/seed identities, metrics and independent receipts. [Editable AN1 Report](necromatcher-arm-morphology-admission-report.tex) and [Report Review Record](historical_capture/arm-morphology-admission-report-review.json) preserve the accepted twelve-page edition. Repository JSON formatting retains the semantic V3 review record; the exact external review SHA and byte count are recorded separately in the compact summary. The report predates final still acceptance; its historical pending language is retained byte-for-byte. This procedure supplies the later accepted image checkpoint.

## Model and Motion Separation

Both players inherit the same generic geometry: authored stature 1.71 m, subject mass 78 kg, arm factor 1.1 and trunk factor 1.15. These are population inputs, not measured player dimensions or standing native height. The 25-body mass sum is 79.3584923989728 kg. Current geometry has 44 scalar coordinates (three metre translations and 41 radian rotations), including 30 upper/neck and 14 leg coordinates. Do not infer topology from the generic model ID or substitute the older upper-body builder.

Public `scale_segments` operates on a deep copy of the exact bound native definition. Relative factors 0.95 and 1.05 change six bilateral upper-arm/proximal-forearm/lower-forearm bodies. Distal translations, centres of mass, placement and declared marker/contact offsets scale linearly; inertia scales quadratically while mass stays fixed. Shoulder hubs, trunk, legs, hands and club remain unchanged. This is an authored fixed-mass approximation, not individualized inertial identification.

Full coordinate order, primitive axes, units, scalar ranges, followers and saved camera remain unchanged. Sparse fit marker offsets are scaled separately by their declared frame identities. Parent subject and geometry provenance are preserved separately; the modified upper-body slice receives its canonical recomputed hash and remains unqualified for original Simscape parity. Authored shaft/grip attachment sites and fixed grip rotation remain unchanged; candidate geometry/axis/proxy identities are rebound through public providers.

Saved-final Hermite coefficients, knot clock and free-coordinate ordering are preserved exactly, with only model identity explicitly rebound. Fresh q/v/a comparisons at source frames and midpoints have zero discrepancy (419 Tiger times and 1,499 Hogan times); strict absolute tolerance is 1e-12 with zero relative tolerance. Preserved coordinates do not imply preserved world positions under changed geometry. No camera or contact phase was silently optimized.

## Actual Results

| Player | Relative Arm Factor | Dense Body RMS (px) | Authored Grip Site Gap (mm) |
| ------ | ------------------- | ------------------- | --------------------------- |
| Tiger  | 0.95                | 23.592441           | 22.057213                   |
| Tiger  | 1.05                | 23.886327           | 21.950475                   |
| Hogan  | 0.95                | 10.055042           | 18.585247                   |
| Hogan  | 1.05                | 10.262778           | 18.641038                   |

Parent dense RMS is 23.582686 px for Tiger and 10.136351 px for Hogan. Parent maximum authored dual-grip world-site gaps are 2.124 mm and 0.733 mm respectively. Hogan's modest minus-five image improvement accompanies substantial grip degradation; none of these results establishes trustworthy subject geometry.

Body RMS uses confidence-weighted Euclidean image distances and unknown-visibility weight 0.5. Each parent retains 286 positive-weight body-training points; source counts remain 210 Tiger and 750 Hogan. Geometry assessment retains 178 Tiger and 364 Hogan nodes with twelve position/rotation/ground row families. The reported maximum scaled component is dimensionless, not a millimetre norm or a continuous constraint certificate.

Shaft training, already-seen V2 evaluation and the unadopted 52-frame diagnostic bucket remain separate. Hogan frame 375 overlaps diagnostic and original training points; it is not a duplicate training term. Abstentions remain unavailable values, not zero residuals. Hogan frame 200 remains approximately 81.700/81.762 degrees from the observed fragment under minus/plus scaling. An accurate direction at frame 375 does not establish the whole trajectory. Authored confidences and sigma values remain uncalibrated, and reviewed images are not unseen validation.

## Persistence and Image Evidence

Native author process 34737 exited zero. Independent serialized persistence process 35753 and root acceptance verify source/runtime preservation, all preexisting non-index bytes and index records, and exactly four model XML plus four seed JSON additions: library 383 to 391. The SDK-free checker did not repeat native calculations. Worker PID/elapsed telemetry was not measured and must not be invented.

Selected-still process 54793 and independent checker 40776 exited zero. Twelve overlays and six unique original frames preserve full source size, encoded PNG hashes, canonical decoded BGR identity and rational source PTS. Source 14,800 files and library 391 files remain unchanged. Root reviewed all twelve selected overlays; this is not all-frame visual acceptance.

Display uses shapes at opacity 0.35, retained blue skeleton, separate authored shaft evidence and compact research captions. It does not change motion, evidence roles or physical qualification. All four admissions remain research hypotheses; fixed-camera/body ambiguity, continuous feasibility, clinical ROM, physical clock and dynamics remain unqualified.

## Reproduction Procedure

1. Authenticate the producer, exact parent fit/model/capture/source clock and immutable PNG identities before invoking SDK-dependent providers. Use the existing public native hypothesis request, binding, authoring and library registration path; never patch stored fit JSON.
2. Reuse the frozen arm generator V2 in `Repositories/Temp/arm-only-hypothesis-preparation-v2`, its checked model documents and the independently accepted admission recipe. Preserve the V1 legacy-candidate failure rather than patching its old-topology assumptions.
3. Use new exclusive model/seed IDs and output destinations for any new attempt. Rebind coordinate units/order, canonical upper-body and shaft/proxy hashes, marker offsets and full q/v/a explicitly before persistence. Do not overwrite these four versions.
4. Reuse the canonical selected-still exporter/controller with frames Tiger 0/150/209 and Hogan 200/375/550. Bracket full source/runtime/library/project before and finally, preserve failures and require closed-process proof before independent checking.
5. Rehash the detailed external report data, raw author results, completion/finally receipts and the independent/root acceptance pins recorded in the compact summary. Copy exact reviewed report source without rebuilding a competing numerical assessment.

External immutable results are under Desktop `Necromatcher Review 2026-10-01/Arm Morphology Native Admission V1`, `Arm Morphology Selected Stills V1` and `Arm Morphology Admission Results Supplement V3`. The local-only 48-payload/49-member offline bundle is reviewed under `Repositories/Temp/arm-morphology-an1-stills-offline-review-v1`; manifest and ZIP pins are in the compact summary. It preserves the historical pre-titlecase V3 report, not the current corrected canonical report. It has not been extracted, transferred to ControlTower or publicly released. Large images/media/ZIP remain outside Git.

The accepted report compiled using existing MiKTeX with installation disabled after the built-in compiler's platform-directory failure. The conjunction-only title-case correction received two installer-disabled passes and page 2 delta review; the other eleven rendered pages are pixel-identical, and the prior exact source/PDF/review remain under `accepted-pre-titlecase-v3`. All twelve final pages were reviewed; zero overfull boxes and two retained underfull lineage-label warnings are documented. Standalone report review does not satisfy engineering manual release: current governance has two QMD sources, zero calculations, and release remains blocked.

## Next Controlled Work

Current user scope excludes Tiger follow-through after right-hand release: subsequent matching and acceptance apply only while both hands remain on the club. Root reviewed twenty selected original images and conservatively retains Tiger frames 0–190 inclusive, the half-open interval [0, 191); uncertain transition and right-hand release are excluded from frame 191 onward. The exact physical release instant is unmeasured, and not all 191 retained frames were visually reviewed. [Source Boundary Review](historical_capture/tiger-two-hand-window-review.json) preserves that decision; its exact external pin is in the summary. Existing fits remain historical whole-capture results. The completed [Conditional Camera Study](necromatcher-conditional-camera-and-scope.md) applies this reviewed scope to fresh comparisons; durable fitting integration remains in progress. Existing whole-capture AN1 measurements remain historical evidence. Released-hand phases must not be counted as failures of the future two-hand acceptance target.

Do not adopt these candidates or increase solver budget automatically. First compare phase-appropriate left-wrist shaft attachment and right-hand closure assumptions, keeping primitive axes distinct from clinical pronation/ROM. More trustworthy phase coverage requires an explicit evidence-role revision: reviewed diagnostic fragments remain unadopted, duplicate Hogan 375 must retain one training identity, and any promoted already-seen labels lose evaluation status.

Joint camera/pose experiments remain distinct from the completed fixed-pose camera perturbations; their poor scores do not rule out coupled camera estimation. Stature/scale and perspective gauge need declared constraints rather than measured-player claims. Coupled clinical ROM requires its own qualified axes and evidence; do not relax authored bounds or closure silently. Rolling readout, exposure, field/raster blur and club flexure are later separate model comparisons, not free absorbers for residuals. Physical timing remains unknown.
