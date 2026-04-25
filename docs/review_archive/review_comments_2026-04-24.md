# Review Comments Archive - 2026-04-24

Generated: 2026-04-24T18:01:35.813203

## Reviewer (chatgpt-codex-connector[bot]) (2 comments)

### PR #3246: tests/examples/test_examples_produce_output.py:37

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Run examples with test interpreter and repo PYTHONPATH**

Spawning examples with `subprocess.run(["python3", str(example_file)])` makes the subprocess import path depend on external shell setup instead of the pytest environment. Running this test from the repo root fails immediately for scripts like `01_basic_simulation.py` with `ModuleNotFoundError: No module named 'src'` because the child process starts wit...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3246#discussion_r3140996269)

---

### PR #3246: tests/examples/test_examples_produce_output.py:49

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Stop requiring stdout for every example script**

This assertion is applied to every discovered file in `examples/*.py`, but several shipped examples are valid run-only demos that currently exit `0` without printing (for example `aerodynamics_demo.py` and `topography_demo.py`). As written, the new parametrized test fails those scripts even when they execute successfully, turning existing behavior into a hard ...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/3246#discussion_r3140996271)

---

