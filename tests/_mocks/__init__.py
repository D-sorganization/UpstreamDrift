"""Shared mock helpers for the Golf Modeling Suite test suite.

Centralising sys.modules mocks here (as opposed to module-level assignments
in individual test files) ensures that each test that needs a stubbed
third-party package opts in explicitly via a fixture. The fixture installs
the stubs through ``monkeypatch.setitem``/``patch.dict``, which auto-clean
after the test completes, avoiding cross-test pollution.
"""
