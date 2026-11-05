---
issue: 14595
summary: "CI Standard sets PIP_RETRIES=10 and PIP_DEFAULT_TIMEOUT=60 workflow-wide and passes pip-audit --timeout 60, so transient PyPI read timeouts no longer fail CI"
---
