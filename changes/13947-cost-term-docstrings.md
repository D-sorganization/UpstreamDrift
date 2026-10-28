---
issue: 13947
summary: "Restore docstrings in motion_matching cost term helpers (UD-20)"
branch: "fix/13947-cost-term-docstrings"
dl_state: "in_review"
next_step: "Merge PR"
---

Move docstring string literals to the beginning of `_position_term`, `_orientation_term`, and `_anchor_term` ahead of local variable assignment in `src/shared/python/motion_matching/cost.py` so functions have valid `__doc__` attributes and do not execute dead string expressions (UD-20, #13947).
