---
issue: 13575
summary: "Retire Jules push/PR noise workflows and use FLEET_GENERAL_AGENT App credentials for fleet policy gates."
---

Removed active Jules automation workflows and aligned surviving App-token workflows with the general fleet agent credentials.

Recorded the deletion of tests/ci/test_jules_push_pr_noise_retirement.py in scripts/config/reviewed_test_deletions.json (#13575) so the merge-queue deleted-test guard accepts the retirement. Replacement coverage is tests/ci/test_workflow_cancellation_semantics.py::test_retired_jules_noise_workflow_not_on_cancel_false_allowlist.
