---
issue: 13949
summary: "Pinocchio viz: adjust 1-based impact_idx to 0-based indexing in error_timecourse (UD-23)"
branch: "fix/13949-pinocchio-viz-impact-idx"
dl_state: "in_review"
next_step: "Merge PR"
---

Adjust 1-based `target.impact_idx` from `ClubTarget` schema to 0-based array index (`impact_idx - 1`) when plotting vertical impact marker lines in `plot_error_timecourse`, bringing Pinocchio viz into parity with Drake, MuJoCo, OpenSim, and Pinocchio's own `fit_swing.py`.
