# API Authentication Baseline Review

The focused camera-preset test independently fails with HTTP 401 instead of
200 (pytest exit 1). This is an existing authentication-fixture mismatch,
not evidence of a shot-pattern physics regression.

`tests/api/test_physics_api.py:286` constructs `TestClient(app)` without an
authentication fixture, dependency override, or bearer header. Its request at
line 299 therefore reaches the global authentication dependency in
`src/api/route_registry.py:185`. Both `GOLF_SUITE_MODE` and
`GOLF_AUTH_DISABLED` are unset. The unchanged environment helper defaults to
`remote` at `src/shared/python/config/environment.py:587`, and the bearer
helper rejects the missing header at `src/api/auth/dependencies.py:231`.
The rejection occurs before the requested endpoint behavior or body validation.

The entire `src/api` tree is unchanged from base
`10c7d61aa686e456b7962f1d9aa9ecfda4e8405f`. The accompanying JSON records
byte-identical SHA-256 comparisons for the test, root test fixtures, server,
router, authentication dependency, physics route, and environment helper.
This is a source-equivalent base reproduction in the current worktree; a
separate full base checkout was not executed. The only `pyproject.toml`
changes add optional report and acceleration extras, not pytest or auth settings.
The preliminary gate's other fourteen 401 failures use the same fixture;
only this representative test was independently rerun.

Reproduction from the worktree root:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy PYTHONPATH=. /tmp/shot-pattern-validation-venv-20261009/bin/python -m pytest tests/api/test_physics_api.py::TestCameraPresetAPI::test_valid_preset_returns_200 -p no:pytestqt -q
```

Follow-up acceptance: make endpoint-behavior tests establish an explicit,
isolated authenticated test context, while retaining separate tests that
assert unauthenticated requests receive 401. Do not weaken application auth
defaults or globally disable authentication to make the suite pass. No API,
auth setting, frozen scientific source, or immutable validation receipt was
changed during this investigation. This finding explains the unrelated gate
failure; it does not waive the repository's pre-PR gate.
