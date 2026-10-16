# F-UD-030

**`np.load(allow_pickle=True)` on candidate `.npz` in two side paths while the canonical loader forbids it**

| Severity | Category | Confidence | Effort | Status   |
| -------- | -------- | ---------- | ------ | -------- |
| ⬜ Low   | security | high       | S      | proposed |

**Locations:** `src/shared/python/motion_matching/visualization/simulation_viewer.py:57`, `src/shared/python/motion_matching/parity_report.py:170`, `src/shared/python/motion_matching/candidate_io.py:103`

## Finding

- Severity: Low - Category: security
- Confidence: high
- Location: src/shared/python/motion_matching/visualization/simulation_viewer.py:57; src/shared/python/motion_matching/parity_report.py:170;
  src/shared/python/motion_matching/candidate_io.py:103
- Evidence: `candidate_io.load_candidate` uses `allow_pickle=False` by design ("immutable numpy arrays without pickle
  dependencies"); `SimulationData.from_npz` and the `parity_report` fallback (reached precisely when `load_candidate` _rejects_
  the file) use `allow_pickle=True`.
- Failure scenario: a malformed/malicious `.npz` that the strict loader refuses is then opened with pickle enabled by the
  fallback — the fallback undoes the hardening. Candidate files are shared between machines and uploaded to leaderboards.
- Why a newer engineer misses this: `allow_pickle=True` is what makes object arrays load; it reads as a compatibility flag.
- Proposed change: `allow_pickle=False` in both; if object arrays are needed, store JSON sidecars (the schema already has them).
- Verification: grep for `allow_pickle=True` in `src/` → 0 (currently 4).
- Effort: S

## Draft Pull Request

**Title:** `fix(motion-matching): load candidate npz files with allow_pickle=False`

### Why This Matters (The Pitch)

candidate_io.load_candidate refuses pickles by design, but SimulationData.from_npz and the parity_report fallback load with allow_pickle=True. The fallback is reached exactly when the strict loader rejects a file, so it undoes the hardening. Candidate files travel between machines and to leaderboards. Switching both to allow_pickle=False, with JSON sidecars if object arrays are needed, is a size-S change.

### Proposed PR Body

## Problem

`motion_matching/visualization/simulation_viewer.py:57` and `parity_report.py:170` call `np.load(..., allow_pickle=True)`, while `candidate_io.py:103` uses `allow_pickle=False` by design. There are 4 occurrences in `src/`.

## Change

- [ ] Use `allow_pickle=False` in both places
- [ ] If object arrays are needed, store JSON sidecars (the schema already has them)

## Verification

`grep` for `allow_pickle=True` in `src/` → 0 (currently 4).

## Risk and Rollback

Low: any file that genuinely needs pickled object arrays will fail to load and must move to sidecars. Rollback is a revert.

## Handoff

None

Refs: adversarial review 2026-10-16, F-UD-030

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
