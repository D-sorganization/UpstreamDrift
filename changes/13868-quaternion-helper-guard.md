---
issue: 13868
summary: "Motion matching: architecture guard and canonical _geodesic helpers (UD-24)"
branch: "fix/13868-quaternion-helper-guard"
dl_state: "in_review"
next_step: "Merge PR"
---

Add architecture guard test tracking private quaternion/geodesic/rotmat helpers outside \_geodesic.py with shrinking baseline allowlist, and provide canonical normalisation, rotmat conversion, and quaternion multiply helpers in \_geodesic.py.
