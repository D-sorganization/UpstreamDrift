---
issue: 14397
title: "Canonical Float Rounding Before Hashing for Ubuntu 26.04"
summary: "physics_spec_sha256 and the saved-comparison receipts compare floats at a canonical precision (12 significant digits, 1e-12 absolute) so BLAS/libm last-bit drift cannot move them (owner option A)"
dl_state: "in_review"
next_step: "Merge the PR."
branch: "agy/14397-ubuntu-2604-hash-drift"
---
