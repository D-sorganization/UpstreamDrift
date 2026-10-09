---
issue: 11787
summary: "Add synthetic sparse inverse-dynamics collocation spike and benchmark existing multiple shooting with fresh replay diagnostics"
dl_state: "in_review"
next_step: "Integrate early F06 native replay validator and D02 frozen private protocol before production backend selection"
branch: "feat/f03-constrained-ocp-11787"
paths: "src/shared/python/motion_matching/sparse_collocation_spike.py,tests/unit/motion_matching/test_sparse_collocation_spike.py,manuals/upstreamdrift/chapters/17-sparse-collocation-benchmark.qmd"
---

Synthetic derivative, hard-bound and fresh-replay tests pass. Input parameterizations differ, so no production solver winner or full-body match is claimed; see docs/development/feedback_controls/F03_TURNOVER.md.
