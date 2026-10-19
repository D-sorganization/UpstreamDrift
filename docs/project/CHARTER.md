# UpstreamDrift Project Charter

## End Goal

Provide reproducible biomechanical golf-swing analysis across multiple physics
engines, with explicit model, data, calibration and numerical provenance. Reuse
shared Tools providers and the existing research/engineering governance described
in [README.md](../../README.md) and [SPEC.md](../../SPEC.md).

This initial feature list covers fourteen deferred external-validation dependencies.
It is not an exhaustive platform inventory, and its feature counts do not measure
whole-project or scientific completion. Executable software and available-data
work remain governed by their original issues and research registrations.

## Non-Goals

- Manufacture measurements, participant results, calibration or independent approval.
- Replace source catalogs, frozen research evidence or qualified provider contracts.
- Treat synthetic tests, issue closure or model fit as physical validation.
- Authorize equipment purchases, participant recruitment or data collection through a status update.

## Features

| ID       | Feature                                                                                                                           | Status | Tracking | Notes                                                                                                                                                                                                                                 |
| -------- | --------------------------------------------------------------------------------------------------------------------------------- | ------ | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DV-11408 | OpenCap: Golf Accuracy Qualification Against a Marker Reference                                                                   | parked | -        | DV-11408: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-11408.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-10382 | MS-108: Observable Golf Outputs, Impact Semantics and Accuracy Budgets                                                            | parked | -        | DV-10382: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-10382.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-10375 | MS-101: Independent Scientific Validation, Identifiability and Uncertainty                                                        | parked | -        | DV-10375: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-10375.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-9700  | Epic: Swing-to-Impact Dynamics and Acoustic Sweetness Research                                                                    | parked | -        | DV-9700: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-9700.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority.   |
| DV-9619  | [EPIC] Self-calibrating markerless mocap: learn camera placement and skeleton from observed swings, with strict outlier rejection | parked | -        | DV-9619: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-9619.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority.   |
| DV-9613  | Rig soak: three-camera sessions intermittently fail to open cameras (dshow I/O error) and a camera re-enumerates mid-series       | parked | -        | DV-9613: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-9613.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority.   |
| DV-9546  | Epic: Deliver and Qualify the Impact Zone and Impact Explorer Product                                                             | parked | -        | DV-9546: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-9546.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority.   |
| DV-11984 | Empirical Qualification of Contact and Delivery Covariance                                                                       | parked | -        | DV-11984: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-11984.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-11542 | MOSAIC-10: Multi-Swing Acquisition and Registration Across Real Golfer Trials                                                    | parked | -        | DV-11542: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-11542.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-12554 | NSX-05: External Validation Protocol for Closed-Loop Null-Space Exploration (Instrumented Grip, Surface EMG, and Held-Out Swings) | parked | -        | DV-12554: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-12554.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-12583 | NSX-55: Golf-Model Simscape Closed-Loop Export in MATLAB R2025b                                                                   | parked | -        | DV-12583: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-12583.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-12718 | NSX-89: Simscape Per-Frame Linearization Export in MATLAB R2025b                                                                  | parked | -        | DV-12718: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-12718.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-13061 | Open-FP Force Plate Hardware Qualification for Measured Vertical GRF                                                              | parked | -        | DV-13061: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-13061.md); fully delegated to DV-13082; Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |
| DV-13082 | Qualified Open-FP Device Record for Measured Vertical Ground Reaction                                                             | parked | -        | DV-13082: [Owner Plan](https://github.com/D-sorganization/UpstreamDrift/blob/main/docs/development/planning/DV-13082.md); Board decision pending; external prerequisites unavailable. Executable work retains its original authority. |

## Source Authority

The [catalog](../development/planning/catalog.json), linked plans and complete
source snapshots retain acceptance and reactivation criteria. Update that single
authority before regenerating this projection. Deferred validation remains an
obligation even when a separate software/product issue has closed. No provider,
engine, calibration or scientific evidence changes are supplied here.
