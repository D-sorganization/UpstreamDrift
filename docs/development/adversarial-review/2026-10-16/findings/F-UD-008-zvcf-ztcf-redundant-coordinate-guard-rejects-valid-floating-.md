# F-UD-008

**ZVCF/ZTCF redundant-coordinate guard rejects valid floating-base states whose last joint is 0**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | S      | proposed |

**Locations:** `src/shared/python/simulation_backends/ztcf_zvcf.py:281`, `src/shared/python/pose_interchange/canonical_layout.py:18`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/simulation_backends/ztcf_zvcf.py:281-300; src/shared/python/pose_interchange/canonical_layout.py:18-23
- Evidence:
  ```python
  canonical_redundant_coordinate = q.shape[0] == tangent.shape[0] + 1 and not np.isclose(q[-1], 0.0)
  ```
  canonical-v2 is `[x, y, z, qw, qx, qy, qz, joints...]`: the redundant quaternion component is at index 3–6, never `q[-1]`.
- Failure scenario (executed): `q = [0,0,1, 1,0,0,0, 0.0]`, `nv = 7` → `PreconditionError: q and zero velocity must share
shape unless q has one populated canonical-v2 redundant coordinate`. Set the last joint to 0.3 and the same call returns.
  Any address pose with the last joint at zero (common: wrist/club joints initialised to 0) cannot be analysed.
- Why a newer engineer misses this: the guard was written to catch "wrong-length zero vector", and `q[-1]` is the cheapest
  thing to test; it happens to be a joint angle, not the quaternion.
- Proposed change: check the quaternion block: `np.isclose(np.linalg.norm(q[3:7]), 1.0)` when `nq == nv + 1`, or accept the
  layout from `canonical_layout.BASE_QUATERNION`.
- Verification: parametrised test over last-joint ∈ {0, 0.3} with a free-flyer provider; both must succeed.
- Effort: S

## Draft Pull Request

**Title:** `fix(ztcf): check the quaternion block, not q[-1], in redundancy guard`

### Why This Matters (The Pitch)

A floating-base address pose with its last joint at zero cannot be analysed. The ZTCF/ZVCF guard decides whether q carries a redundant quaternion coordinate by testing that q[-1] is not close to 0, but in canonical-v2 the quaternion sits at indices 3 to 7, and q[-1] is just a joint angle. Executed: q = [0,0,1, 1,0,0,0, 0.0] with nv = 7 raises PreconditionError; set the last joint to 0.3 and the same call returns. Wrist and club joints initialised to zero are common, so the primitives that the owner most wants other people to run reject a routine input. The fix is to check the quaternion norm (or use the layout from canonical_layout.BASE_QUATERNION), and a parametrised test over last-joint values 0 and 0.3 pins it. Size S in an otherwise reference-quality file.

### Proposed PR Body

## Problem

`src/shared/python/simulation_backends/ztcf_zvcf.py:281-300` computes `canonical_redundant_coordinate = q.shape[0] == tangent.shape[0] + 1 and not np.isclose(q[-1], 0.0)`, but canonical-v2 (`pose_interchange/canonical_layout.py:18-23`) is `[x, y, z, qw, qx, qy, qz, joints...]`. With `q = [0,0,1, 1,0,0,0, 0.0]` and `nv = 7` it raises `PreconditionError`; with the last joint 0.3 it succeeds.

## Change

- [ ] Replace the `q[-1]` test with `np.isclose(np.linalg.norm(q[3:7]), 1.0)` when `nq == nv + 1`, or read the layout from `canonical_layout.BASE_QUATERNION`

## Verification

Parametrised test over last-joint ∈ {0, 0.3} with a free-flyer provider; both must succeed.

## Risk and Rollback

Low: widens accepted inputs to those the layout defines as valid. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-008

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
