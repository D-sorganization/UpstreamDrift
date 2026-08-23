# Launch-Mode Functional QA Gate

Issue [#8966](https://github.com/D-sorganization/UpstreamDrift/issues/8966)
(EPIC #8965, WS1). Proves each launch mode routed by
`launch_upstream_drift.py` reaches a ready state, so regressions like
#8852/#8854/#8860 cannot ship silently again.

## Running

```bash
pytest tests/launch_modes -m launch_qa
```

The tests live in `tests/launch_modes/test_launch_matrix.py`, carry the
`launch_qa` marker (registered in `pyproject.toml`), and run inside the
normal pytest suite — no workflow changes required. Deselect with
`-m "not launch_qa"`. Qt runs offscreen (`QT_QPA_PLATFORM=offscreen`);
no windows open and no subprocesses are spawned.

The classic-mode and parity tests need the `vendor/ud-tools` submodule
checked out (`git submodule update --init vendor/ud-tools`), matching CI.

## The Matrix

| Mode              | What is asserted                                                                                                                                     | Ready signal                                                  |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| web (default)     | `create_local_app()` builds within budget; health endpoint answers; manifest serves a non-empty, unique tile set                                     | `GET /api/health` 200, `GET /api/launcher/manifest` tiles > 0 |
| `--classic`       | `UpstreamDriftLauncher` constructs offscreen with a **real** `ModelRegistry`; tile grid populated; no unhandled exception                            | `available_models` and `model_cards` non-empty                |
| `--api-only`      | The same app object serves the interactive API docs                                                                                                  | `GET /api/docs` 200                                           |
| `--engine <id>`   | The exact module `launch_engine_directly` imports resolves and exposes `main()`; MuJoCo additionally constructs its launcher window without `exec()` | import + `main` callable; window has a central widget         |
| parity            | Web manifest tile IDs equal the PyQt registry's model IDs                                                                                            | set equality                                                  |
| DbC postcondition | Every ready-status tile's `path` resolves on disk                                                                                                    | no dead targets                                               |

## Budget Policy

- Web app construction must finish within **60 s**
  (`WEB_APP_CONSTRUCTION_BUDGET_S` in `tests/launch_modes/conftest.py`).
  Deliberately generous cold-start ceiling (ties to #8934/#8938) —
  ratchet down as startup work lands.
- Classic construction and the MuJoCo window test carry hard
  `pytest.mark.timeout` guards so a hang fails instead of wedging CI.

## Skip / Xfail Policy — Never Fake Success

- An engine whose runtime is not installed **skips** with
  `engine not installed: <id>`. Web-only engines (`matlab_2d`,
  `matlab_3d`) skip because they route to the web UI, which the web
  tests already cover.
- Known-broken behavior **xfails carrying an issue number** and flips
  to pass when the fix lands (`strict=False` semantics):
  - `#8853` — web manifest vs PyQt registry are two tile registries.
  - `#8854` — ready-status tiles with unresolvable target paths.
  - `#8967` — `--engine mujoco` module lacks `main()` /
    `HumanoidLauncher.__init__` TypeError; `--engine pendulum` module
    missing.
  - `#8972` — classic `init_ui` imports theme names that do not exist.
- Anything else failing is a real regression: fix the launch path, do
  not widen the skip list.
