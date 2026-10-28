---
issue: 13971
summary: "Drake motion matching: catalog finite-difference generalized accelerations in simulate_with_coefficients (UD-36)"
branch: "test/13971-catalog-finite-difference-qdd-drake"
dl_state: "in_review"
next_step: "Merge PR"
---

Catalog contract that Drake's `simulate_with_coefficients` populates generalized accelerations `SimOut.qdd` using discrete central finite differences via `finite_difference_qdd` sampled from the velocity trajectory, with continuous MultibodyPlant acceleration output port sampling tracked under Epic #13789.
