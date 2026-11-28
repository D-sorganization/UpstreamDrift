---
issue: 15497
summary: "Explain missing bunker flight and green rollout traces at the model boundary"
dl_state: "in_review"
next_step: "Independent review and CI; leave PR unarmed per owner instruction"
branch: "fix/15497-flight-unavailable"
---

The headless model now records an explicit unavailable reason when a carry
estimate produces no usable flight/settled-green trace, including a missing
retained launch. Existing adapter exceptions retain their original reason
without a second generic diagnostic. Valid traces and configured timesteps
pass through unchanged. No trajectory is fabricated or physics changed.

TDD: two omission cases failed before the production change; seven focused
unit cases cover omission, successful forwarding, recoverable adapter errors,
and unavailable carry. The flight service boundary uses synthetic traces;
these tests establish orchestration behavior, not native physics acceptance.
