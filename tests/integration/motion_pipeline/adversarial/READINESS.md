# Motion Pipeline — Functional Readiness Report

> Adversarial review conducted 2026-05-08 on `origin/main` (Python 3.13).
> See PR for full context.

## Confidence rating: **YELLOW**

The CIR contract layer and adapter loaders are solid. Preprocessing,
scaling, IK, and matching primitives are usable for pure-Python
unit-test workloads. **However, the HTTP API is currently broken and
the documented CLI surface is fictional** — both block end-to-end
real-data integration.

## Module status

| Module | Status | Notes |
|---|---|---|
| `contracts.py` | GREEN | 93.5% coverage. Most invariants enforced. Two gaps filed (#4720). |
| `sources/` (loaders) | YELLOW | All 9 formats import; 5/9 golden roundtrips fail on main (pre-existing). c3d adapter leaks raw OSError (#4721). |
| `preprocessing/` | YELLOW | Pipeline imports cleanly. Kalman filter declared but unimplemented (existing xfail). PCA gap-fill missing (existing xfail). |
| `scaling/` | YELLOW | anthropometric works; opensim_scale 13.87% covered (deps missing). |
| `ik/` | GREEN | All 4 backend stubs import; base class 83% covered. |
| `matching/` | YELLOW | base + costs work; inverse_dyn_pinocchio 52% covered, opensim variants skipped. |
| `orchestrator.py` | YELLOW | 44% coverage. Several wiring gaps tracked (#4647/#4648/#4649). |
| `api.py` | **RED** | `create_app()` crashes at registration time (#4722). HTTP API is non-functional. |

## Real-world data scenarios exercised

The adversarial suite (`tests/integration/motion_pipeline/adversarial/`)
exercises:

- **Malformed input rejection** for every format extension (empty,
  truncated, wrong-format, garbage-bytes, NaN/Inf, negative timestamps,
  duplicate timestamps, oversized zero-padded, Unicode names).
- **Boundary conditions:** 1-frame, 1-joint, all-zero confidence,
  inverted joint limits, locked joint, multi-target requirement.
- **CIR invariant matrix** parametrised over 7 (model, bad_kwargs)
  pairs; xfailed cases mark filed bugs.
- **Determinism:** `model_dump_json()` byte-equality across repeated
  construction and roundtrip.
- **Real-world schema drift:** MediaPipe `landmarks` vs
  `pose_landmarks`, OpenPose BODY_25 vs COCO_17, BVH ZYX/XYZ/ZXY
  rotation channels, TRC blank-line tolerance.
- **Concurrency safety:** 4-thread CIR construction byte-equality;
  4-thread loader no-crash.
- **Resource bounds:** 60000-frame KeypointSequence build under 60s
  (slow-marked).
- **LoD audit:** every `motion_pipeline/**/*.py` is checked to import
  zero GUI toolkits and zero outbound-network libraries.

## Issues filed in this review

| Issue | Title | Severity |
|---|---|---|
| [#4720](https://github.com/D-sorganization/UpstreamDrift/issues/4720) | JointDef accepts empty name and non-3D offset | HIGH (silent contract bypass) |
| [#4721](https://github.com/D-sorganization/UpstreamDrift/issues/4721) | C3D adapter raises raw OSError on empty file | MEDIUM (typed-exception leak) |
| [#4722](https://github.com/D-sorganization/UpstreamDrift/issues/4722) | FastAPI create_app() crashes — entire HTTP API non-functional | **CRITICAL** |
| [#4723](https://github.com/D-sorganization/UpstreamDrift/issues/4723) | README advertises 5 CLI commands that do not exist | HIGH (user-facing) |

## Pre-existing baseline (origin/main)

449 passed, 22 failed, 91 skipped, 5 xfailed, 2 xpassed, 4 errors.
Most failures cluster in matching/inverse_dyn_pinocchio, contracts edge
cases, and 5/9 golden-roundtrip fixtures. None of these are caused by
this review's additions.

## Confidence call: ready for real markerless mocap data?

**No — Conditional.** Conditions:

1. **#4722 must land first.** Without a working HTTP API the orchestrator
   cannot be exercised by external mocap tools.
2. **#4720 must land before any real skeleton ingest.** A malformed
   `JointDef` with `len(offset) == 2` will silently propagate and explode
   only at IK time with an opaque NumPy broadcast error.
3. **#4723 must land before any new contributor onboards.** The README
   is the front door; today it points users at modules that don't exist.
4. The 22 pre-existing test failures should be triaged. Many are
   surface-level (e.g., NaN focal length validation) but a few
   (matching contract roundtrip, scaling 1.5x recovery) suggest deeper
   numerical issues that will surface as soon as real data appears.

Once #4720 / #4721 / #4722 / #4723 are closed, the rating will move to
GREEN.
