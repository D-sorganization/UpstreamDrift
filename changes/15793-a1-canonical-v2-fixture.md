---
issue: 15793
summary: "Freeze MuJoCo canonical-v2 / capture-A reference fixture schema and integrity checks (A1)"
dl_state: "in_review"
next_step: "Proceed to Slice A2 skeleton dimension audit and prescribed kinematic replay against the frozen fixture."
branch: "ag/15793-a1-canonical-v2-fixture"
---

Added `src/engines/physics_engines/opensim/python/motion_matching/canonical_v2_reference_fixture.py` with schema `opensim-canonical-v2-fixture/1.0.0`, `FrameTransform3D` SO(3) validation, `CanonicalV2ReferenceTrajectory` monotone/finite timebase validation, `CanonicalV2ReferenceFixture` deterministic JSON serialization and SHA-256 tamper detection, and fail-closed `require_canonical_golfer` gating, verified by TDD unit tests in `tests/unit/engines/opensim/test_canonical_v2_reference_fixture.py`.
