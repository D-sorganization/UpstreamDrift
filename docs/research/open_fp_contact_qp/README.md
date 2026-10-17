# Measured Plate Rows in the Contact-Wrench QP

Start with the [research proposal](proposal.tex) for OF-10 (#13089) under epic
#13061. It is an editable standalone research source (see `AGENTS.md`, Modeling
Reference Documentation). It changes no code. Implementation is gated on OF-08
(#13086) and on hardware qualification OF-HW (#13082).

| Item                                                     | Status                       |
| -------------------------------------------------------- | ---------------------------- |
| Exact linear rows for load and sensing-plane CoP         | Derived in the proposal      |
| Weighting from the OF-05 budget; clock offset profiled   | Proposed                     |
| Band constraints with reported slack; OF-12 screen first | Proposed                     |
| Held-out-frame and held-out-trial validation             | Proposed                     |
| Solver change in `contact_wrench_qp.py`                  | **Not started** (gated)      |
| Evidence from a qualified device                         | **Deferred** (OF-HW, #13082) |
