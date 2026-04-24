---
name: browser-journey-test
description: Validate user stories by simulating real user journeys in a live browser against deployed application. Activate after implementation to verify actual user experience against acceptance criteria, not just code logic.
license: MIT
compatibility: Works with any filesystem-based AI coding agent (requires browser automation capability)
metadata:
  author: gaai-framework
  version: "1.1"
  category: delivery
  track: delivery
  id: SKILL-BROWSER-JOURNEY-TEST-001
  updated_at: 2026-02-26
  status: experimental
  required_capability: browser-automation
inputs:
  - contexts/artefacts/stories/**
  - deployed_application_url # Provided by the invoking agent from the staging deploy output. Not supplied manually.
outputs:
  - contexts/artefacts/test-evidence/{story_id}/journey-test-report.md
---

# Browser Journey Test

## Purpose / When to Activate

Activate after implementation to validate real user experience — not just code logic.

Use when:
