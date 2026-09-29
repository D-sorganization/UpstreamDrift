"""Adversarial integration tests for the motion pipeline.

These tests intentionally try to break the pipeline. Tests that surface a
production bug are marked ``xfail(strict=True)`` with a cross-reference to
a filed GitHub issue. Tests that pass demonstrate hardening that has
already shipped.
"""
