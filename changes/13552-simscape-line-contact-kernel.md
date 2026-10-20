---
issue: 13552
summary: "Simscape grip kernel follows the shared capsule line-contact law; ten_finger verifier back to 1e-11; structural freshness check for generated JSON; committed R2025b evidence"
dl_state: "in_review"
next_step: "Merge via queue. Any later change to the shared grip law or scenarios requires regenerating the kernel (scripts/export_simscape_grip_kernel.py) and rerunning R2025b on DeskComputer; test_r2025b_receipt_hashes_match_committed_kernel_and_inputs fails until the receipts are refreshed."
branch: "fix/13334-simscape-line-contact-kernel"
---

Kernel regenerated from main's law (#13457/#13508); R2025b rerun 2026-10-20 on DeskComputer: golden 51/51, 81 compiled blocks, net-club-wrench parity vs Pinocchio worst 0.011 % peak / 0.045 % RMS. Known pre-existing failures: Pinocchio swing replay (fails on clean main); MyoHand law-level spin channel at 1e-14 N·m when myo_sim is populated (follow-up queued). Open follow-up: t=0 lead heel-pad 42.77 N preload transient in overlap/interlock.
