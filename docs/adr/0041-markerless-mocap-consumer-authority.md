# ADR 0041: Markerless Mocap Consumer Authority

- **Status:** Accepted for consumer-boundary implementation
- **Date:** 2026-08-25
- **Issues:** #9063, #9065, #9069; Tools #4706, #4708, #4710
- **Validation:** `tests/architecture/test_markerless_mocap_authority.py`

## Context

UpstreamDrift already reads C3D, imports OpenCap-style results, runs a separately
installed FreeMoCap sidecar, maps motion targets, and presents C3D data. Closed
work under #4558 and related children established file-based ingestion and C3D
output. Open issue #8865 records two competing C3D ingestion stacks and a GUI
path that bypasses the canonical motion pipeline. None of that work defines a
live, arbitrary-camera lab or authorizes a second set of camera, observation,
calibration, or reconstruction records in this repository.

Tools #4571 concerns cameras in a simulation viewport. It is unrelated to
physical camera discovery, synchronization, calibration, or markerless motion
capture. The two programs must not share issue status or qualification claims.

## Decision

One repository owns each responsibility.

| Authority     | Owned Responsibilities                                                                                                                                                                                                           |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tools         | MIT vendor-neutral camera and capture contracts, time and synchronization evidence, calibration records, 2-D observations, 3-D reconstruction records and reference algorithms, session interchange, and C3D exchange contracts. |
| UpstreamDrift | Session orchestration, project persistence, operator workflows, APIs, PyQt6 and React/Tauri UX, units and theme presentation, contextual help, biomechanics and motion-matching adapters, and C3D workflow integration.          |
| AffineDrift   | Sanitized evidence publication, validation dossiers, pedagogical workbenches, and immutable public projections.                                                                                                                  |
| Tools_Private | Private operational tools only; it is not part of the public runtime and cannot be required by an open markerless-mocap installation.                                                                                            |

UpstreamDrift will consume the public `sidekick.lab.mocap` boundary after the
Tools candidate is protected-merged and `vendor/ud-tools` is pinned to that
merge. No feature-branch Tools commit is release authority. The first product
adapter belongs to UpstreamDrift #9069 and must reject an absent or incompatible
schema instead of copying Tools-owned source below `src/shared/python`.

## Coordinate and Timing Boundary

The candidate shared world frame is right-handed SI: x toward the target, y up,
and z right. A camera transform is named by direction,
`T_world_from_camera`; a generic `extrinsics` field is prohibited. With camera
intrinsics `K`, rotation `R`, translation `t`, and homogeneous world point
`X_world`, projection is written explicitly as

`s [u, v, 1]^T = K [R_camera_from_world | t_camera_from_world] X_world`.

Every adapter must prove which directed transform it supplies before applying
that equation. Device, trigger, host-monotonic, and UTC presentation clocks are
different evidence domains. Arrival time is not silently promoted to exposure
time. Timing uncertainty and dropped-frame provenance remain attached to the
observations that use them.

## Evidence and Reconstruction Boundary

Multi-view reconstruction requires at least two unique, calibrated useful
views and records contributing and rejected cameras, residuals, and
uncertainty. Single-camera depth inferred by a learned prior is
`model-conditioned`; it is not labelled triangulated or observed 3-D.
Camera-agnostic means providers negotiate explicit supported, degraded, or
unsupported capabilities. It does not mean all shutters, frame rates,
synchronization modes, lenses, or placements are scientifically equivalent.

Acceptance depends on observability, useful views, residuals, timing, and
uncertainty rather than a fixed camera count. Body pose and high-speed
club/ball capture may use different devices and inference paths while sharing
session identity, time, coordinates, and provenance.

## Licensing and Privacy Boundary

The open core must remain redistributable under its declared licenses. AGPL
applications such as FreeMoCap or SkellyCam may be used only as separately
installed and separately started subprocess or IPC services after legal and
distribution review. They are not imported, vendored, linked, bundled, or
declared as required runtime dependencies of the MIT Tools core. Restricted or
noncommercial model weights and body models are never the commercial default.

Raw video is sensitive. Capture requires recorded consent, explicit retention,
no-store behavior where requested, bounded storage, and visible operator state.
Missing consent or required production authentication fails closed; synthetic
data is not substituted for absent governed recordings.

## C3D Boundary

C3D export must adapt the canonical Tools observation/session contract through
one characterized UpstreamDrift workflow. It must preserve point units, sample
rates, coordinate direction, residual/missing-point semantics, labels, events,
and analog timing where present. The existing reader, Rust parser, exporter,
viewer, and motion-pipeline adapter are compatibility evidence to consolidate
under #8865, not permission to add another reader.

## Consequences

- M0 supplies governance and executable drift detection, not live capture.
- Tools #4706 must merge before the pinned product consumer is implemented.
- UpstreamDrift owns friendly recovery and help but never invents provider
  capability, calibration, or reconstruction authority.
- Synthetic fixtures qualify software contracts only. They do not qualify a
  camera, shop layout, person, biomechanical conclusion, or commercial lab.

## Validation

- Architecture tests pin ownership, license/privacy, existing-work, C3D, and
  qualification language.
- Title-case, file-size, SPEC, and handoff gates apply to this decision.
- Later slices add exact Tools-SHA consumer tests, schema migration goldens,
  PyQt/React/API parity, faults, clean-export, and physical acceptance evidence.
