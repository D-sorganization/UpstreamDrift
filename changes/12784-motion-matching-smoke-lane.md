---
issue: 12784
summary: "Nightly motion-matching-smoke.yml runs tests/integration/motion_matching (slow, --timeout=0) via scripts/ci/run_motion_matching_smoke.sh on the fleet; a failure opens or updates an issue and the red scheduled run reaches fleet-main-health; not coupled to heavy-tests-opt-in parity"
dl_state: "in_review"
next_step: "Merge the PR; the first scheduled run needs #12776 and #12783 on main to be green."
branch: "ci/12784-motion-matching-smoke-lane"
---
