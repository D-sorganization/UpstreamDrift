---
issue: 12702
summary: "Fix the 'spec has no club body' crash on the MuJoCo-native driver run: assembly_from_spec now recognises the native Simscape club (segmented 2.5in Butt grip, no Grip solid) so the default v2 spec carries a club body; Lane.set_club_targets fails fast via club_face_target.require_club_assembly naming the spec source. The smoke test now gets past set_club_targets but still fails on dynamics marker RMS (0.51 m vs 0.15 m bound), a separate regression; the smoke is marked slow and runs in no CI lane"
dl_state: "in_review"
next_step: "Merge the PR; triage the driver dynamics marker RMS regression (0.089 m measured 2026-09-30, 0.51 m now, independent of --face-weight) and put the slow smoke in a nightly lane."
branch: "fix/12702-mujoco-smoke-club-body"
---
