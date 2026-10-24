---
issue: 13552
summary: "Simscape grip kernel follows the shared capsule line-contact law; ten_finger verifier back to 1e-11; structural freshness check for generated JSON; goldens rebuilt from the rounded export params; committed R2025b evidence"
dl_state: "in_review"
next_step: "Merge via queue. Any later change to the shared grip law or scenarios requires regenerating the kernel (scripts/export_simscape_grip_kernel.py) and rerunning R2025b (gs3dx_grip_contact_run); test_r2025b_receipt_hashes_match_committed_kernel_and_inputs fails until the receipts are refreshed."
branch: "claude/grip-13556-rounded-goldens"
---

Kernel regenerated from main's law (#13457/#13508); R2025b rerun 2026-10-20 on DeskComputer: golden 51/51, 81 compiled blocks, net-club-wrench parity vs Pinocchio worst 0.011 % peak / 0.045 % RMS. Known pre-existing failures: Pinocchio swing replay (fails on clean main); MyoHand law-level spin channel at 1e-14 N·m when myo_sim is populated (follow-up queued). Open follow-up: t=0 lead heel-pad 42.77 N preload transient in overlap/interlock. Goldens are now computed from the same 11-decimal canonicalized params that are exported (owner decision 2026-10-23); the 1e-11 law cross-check uses the unrounded params. R2025b rerun 2026-10-23 on ControlTower: golden 51/51 cases, 830/830 verifications, receipts regenerated.
