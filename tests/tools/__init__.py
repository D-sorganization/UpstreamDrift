"""Tests for the standalone tools under ``src/tools``.

This file makes ``tests/tools`` a regular package, so pytest imports each test
package here as ``tests.tools.<name>``. Without it, pytest puts ``tests/tools``
itself on ``sys.path`` and every test package below claims the same top-level
name as the ``src/tools`` package it tests (#7995, #8053).
"""
