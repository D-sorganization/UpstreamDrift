# F-UD-017

**Target resampling extrapolates by constant hold and never masks it; cost then scores fabricated frames**

| Severity  | Category | Confidence | Effort | Status   |
| --------- | -------- | ---------- | ------ | -------- |
| 🟨 Medium | numerics | high       | M      | proposed |

**Locations:** `src/shared/python/motion_matching/loaders/_align.py:85`, `src/shared/python/motion_matching/_series_interp.py:14`, `src/shared/python/motion_matching/cost.py:171`

## Finding

- Severity: Medium - Category: numerics
- Confidence: high
- Location: src/shared/python/motion_matching/loaders/\_align.py:85-96; src/shared/python/motion_matching/\_series_interp.py:14-21;
  src/shared/python/motion_matching/cost.py:171-178
- Evidence: `interp_xyz_series` is `np.interp` ("endpoint clamp"); `resample_target` shifts `raw_time` so impact lands at
  `impact_target_t_s` and samples `0 : 1/fs : T` regardless of capture extent; `_position_term` is `mean_n` over all N with no
  validity mask; `np.interp` is not told whether `raw_time` is increasing.
- Failure scenario: a capture that starts 0.1 s after the sim window starts, or ends before `simulation_time_s`, contributes
  frozen clubhead positions that the simulation is penalised for not matching — biasing `theta` toward "stop moving at the
  edges". A non-monotonic C3D timestamp (dropped frames re-ordered) makes `np.interp` return garbage silently.
- Why a newer engineer misses this: `np.interp` is convenient and its clamping is documented; the omission is the mask that
  downstream never asks for.
- Proposed change: return a `valid: (N,) bool` alongside the resampled arrays (`raw_time[0] <= t <= raw_time[-1]`), store it on
  `ClubTarget`, and have `_position_term/_orientation_term` average only valid rows; `require(np.all(np.diff(raw_time) > 0))`.
- Verification: a target covering 50 % of the window must give the same `J` as the same target zero-padded with mask=False.
- Effort: M

## Draft Pull Request

**Title:** `fix(motion-matching): mask out-of-range target samples in the cost`

### Why This Matters (The Pitch)

When a motion capture clip does not cover the simulation window, the resampler clamps to the endpoint values and the cost then scores those fabricated frames. A capture that starts 0.1 s after the sim window, or ends before simulation_time_s, contributes frozen clubhead positions that the simulation is penalised for not matching, biasing theta toward 'stop moving at the edges'. Separately, np.interp is not told the timestamps are increasing, so a non-monotonic C3D timestamp from re-ordered dropped frames silently yields garbage. Nobody sees it because np.interp's clamping is documented and convenient; what is missing is the validity mask that downstream never asks for. Returning a valid boolean array from resample_target, storing it on ClubTarget, averaging only valid rows and requiring monotonic time makes the fit honest about data it does not have, which matters to anyone whose scientific claims rest on the fitted parameters.

### Proposed PR Body

## Problem

`src/shared/python/motion_matching/loaders/_align.py:85-96` resamples via `interp_xyz_series` (`_series_interp.py:14-21`, `np.interp` endpoint clamp) onto `0 : 1/fs : T` regardless of capture extent; `cost.py:171-178` `_position_term` averages over all N with no validity mask, and monotonicity of `raw_time` is never checked.

## Change

- [ ] Return `valid: (N,) bool` (`raw_time[0] <= t <= raw_time[-1]`) alongside resampled arrays and store it on `ClubTarget`
- [ ] Make `_position_term`/`_orientation_term` average only valid rows
- [ ] `require(np.all(np.diff(raw_time) > 0))`

## Verification

A target covering 50 % of the window must give the same `J` as the same target zero-padded with mask=False.

## Risk and Rollback

Medium: cost values for partially covered targets will change (that is the fix), so golden values may need regenerating. Rollback is a revert.

## Handoff

- **Branch, commit, and pull request:** `<branch>` @ `<commit>` — PR #<n> (fill in when opened)
- **Validation:** the command(s) under _Verification_ above, run locally and green; plus the repo's fast gates
  (`ruff check`, `ruff format --check`, the relevant `-m unit` subset)
- **Blockers:** none known at drafting time — record any that appear
- **Next step:** open the PR against `main`, arm auto-merge with `scripts/automerge_guard.py`, end the session
- **Follow-ups:** none

Refs: adversarial review 2026-10-16, F-UD-017

---

_Generated 2026-10-16 from an adversarial review of `UpstreamDrift` @ `87d6d6ee`. This is a proposal, not a verdict — verify the cited lines before acting._
