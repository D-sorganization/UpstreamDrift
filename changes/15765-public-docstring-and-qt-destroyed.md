---
issue: 15765
summary: "Code-quality docstrings follow the public API, and Qt destroyed callbacks no longer close over self"
---

The fail-closed checker from #15765 was walking every function in a changed
file, so private helpers, nested functions, and tests failed CI after a local
ruff run stayed green. Public sync functions, public async functions, and
public methods still require a docstring.

Three `destroyed` lambdas loaded `self` from a method cell. Teardown clears
that cell and pytest-qt fails whichever test is on the worker. The callbacks
now bind the widget by value.
