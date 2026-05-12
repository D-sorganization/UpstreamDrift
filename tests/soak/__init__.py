"""Long-running soak tests, gated on ``SOAK=1`` env var.

These tests are excluded from normal CI by default; the gating happens at
collection time inside each test module so a stray ``pytest tests/soak``
invocation in a normal environment exits with skip, not failure.
"""
