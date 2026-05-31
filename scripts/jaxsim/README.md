# JaxSim Scripts

Helper scripts for the JaxSim backend (see
[`src/engines/physics_engines/jaxsim/README.md`](../../src/engines/physics_engines/jaxsim/README.md)).

| Script                          | Purpose                                                                                                                                                                                                                                  |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `check_jaxsim_pin.py`           | Upgrade-guard (issue #6660): fails if the `jaxsim` pin drifts from `==0.9.0` in `pyproject.toml` or in the installed stack. Runs as a CI step in `cross-engine-equivalence.yml` and is covered by `tests/unit/test_jaxsim_pin_guard.py`. |
| `plot_parameter_sensitivity.py` | Renders a deterministic ZTCF parameter-sensitivity plot (issue #6656, gated). Requires the optional JaxSim stack.                                                                                                                        |

## Upgrade guard

```bash
python scripts/jaxsim/check_jaxsim_pin.py
```

Exit code `0` means the pin is intact; non-zero means it drifted. JaxSim is
intentionally pinned while the integration is gated — coordinate any bump with
the cross-engine parity and forward-sim gates.
