# Third-Party License Ledger

Issue: [#6781](https://github.com/D-sorganization/UpstreamDrift/issues/6781)

This ledger is the commercialization gate for direct runtime and optional
dependencies declared in `pyproject.toml`. The "Version basis" column records
the declared constraint because the repo currently uses range constraints for
most dependencies rather than one fully pinned application lock. Re-run
`python scripts/legal/check_license_ledger.py` after dependency changes.

Status meanings:

- Commercial-OK: permissive or normally commercial-compatible license.
- Needs review: verify current package metadata, bundled assets, model weights,
  or platform terms before commercial distribution.
- Non-commercial: do not ship in the commercial default path.
- Opt-in: available only when a user deliberately enables that integration.

## Commercial Motion-Capture Path

MediaPipe is the default commercial-clear markerless detector path for local
video ingestion. OpenPose remains an optional external integration because its
upstream project has historically carried non-commercial/research-use terms and
separate model/distribution requirements. Any new ingestion adapter should
default to MediaPipe or another permissively licensed detector and document the
model/backbone license separately.

## Direct Dependency Ledger

| Package                  | Scope                            | Version basis                    | License                               | Status                 | Notes                                                                        |
| ------------------------ | -------------------------------- | -------------------------------- | ------------------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `alembic`                | dev                              | `>=1.13.0`                       | MIT                                   | Commercial-OK          | Migration tooling.                                                           |
| `anyio`                  | dev                              | `>=4.0.0`                        | MIT                                   | Commercial-OK          | Async test dependency.                                                       |
| `bcrypt`                 | dev                              | `>=4.1.0`                        | Apache-2.0                            | Commercial-OK          | Auth/security tests.                                                         |
| `bokeh`                  | core                             | `>=3.8.2`                        | BSD-3-Clause                          | Commercial-OK          | Visualization runtime.                                                       |
| `build`                  | dev                              | unpinned                         | MIT                                   | Commercial-OK          | Packaging helper.                                                            |
| `cryptography`           | dev                              | `>=46.0.7`                       | Apache-2.0 OR BSD-3-Clause            | Commercial-OK          | Security tests.                                                              |
| `defusedxml`             | urdf, dev                        | `>=0.7.0`                        | Python-2.0                            | Commercial-OK          | XML hardening.                                                               |
| `drake`                  | drake                            | `>=1.22.0,<2.0.0`                | BSD-3-Clause                          | Commercial-OK          | Optional engine; verify bundled solver terms on upgrade.                     |
| `email-validator`        | dev                              | `>=2.0.0`                        | Unlicense                             | Commercial-OK          | Pydantic email validation.                                                   |
| `ezc3d`                  | dev                              | `>=1.5.0`                        | MIT                                   | Commercial-OK          | C3D test dependency.                                                         |
| `fastapi`                | core                             | `>=0.126.0`                      | MIT                                   | Commercial-OK          | API framework.                                                               |
| `flask`                  | core                             | `>=3.1.3`                        | BSD-3-Clause                          | Commercial-OK          | Runtime security floor dependency.                                           |
| `gymnasium`              | biomechanics, experimental, rl   | `>=0.29.0`                       | MIT                                   | Commercial-OK          | RL/biomechanics environment API.                                             |
| `h5py`                   | core                             | `>=3.0.0`                        | BSD-3-Clause                          | Commercial-OK          | HDF5 IO.                                                                     |
| `httpx`                  | core                             | `>=0.27.0`                       | BSD-3-Clause                          | Commercial-OK          | HTTP client.                                                                 |
| `hypothesis`             | dev                              | `>=6.0.0`                        | MPL-2.0                               | Commercial-OK          | Test-only property testing; keep out of distributed runtime unless reviewed. |
| `imageio`                | pose                             | `>=2.31.0`                       | BSD-2-Clause                          | Commercial-OK          | Video/image IO; ffmpeg binary terms depend on installed build.               |
| `jax`                    | mjx                              | `>=0.4.30`                       | Apache-2.0                            | Commercial-OK          | JAX runtime for MJX differentiable simulation.                               |
| `jaxlib`                 | mjx                              | `>=0.4.30`                       | Apache-2.0                            | Commercial-OK          | Compiled JAX runtime for MJX differentiable simulation.                      |
| `jinja2`                 | core                             | `>=3.1.0`                        | BSD-3-Clause                          | Commercial-OK          | Template rendering.                                                          |
| `jaxsim`                 | jaxsim                           | `==0.9.0`                        | BSD-3-Clause                          | Commercial-OK          | Optional engine.                                                             |
| `lxml`                   | dev                              | `>=5.0.0`                        | BSD-3-Clause                          | Commercial-OK          | URDF schema validation tests.                                                |
| `matplotlib`             | dev, gui-tools, body-part-viz-gl | `>=3.10.8`                       | PSF-based                             | Commercial-OK          | Plotting/test dependency.                                                    |
| `mediapipe`              | pose                             | `>=0.10.33`                      | Apache-2.0                            | Commercial-OK          | Default markerless pose detector path.                                       |
| `meshcat`                | pinocchio                        | `>=0.3.0,<1.0.0`                 | MIT                                   | Commercial-OK          | Optional Pinocchio visualization.                                            |
| `mujoco`                 | core                             | `>=3.6.0,<4.0.0`                 | Apache-2.0                            | Commercial-OK          | Default physics engine.                                                      |
| `mujoco-mjx`             | mjx                              | `>=3.6.0`                        | Apache-2.0                            | Commercial-OK          | MuJoCo MJX optional differentiable backend.                                  |
| `mujoco-warp`            | warp                             | `>=0.1.0`                        | Apache-2.0                            | Commercial-OK          | Optional GPU backend; verify NVIDIA Warp transitive terms.                   |
| `mypy`                   | dev                              | `>=1.20.1`                       | MIT                                   | Commercial-OK          | Type checker.                                                                |
| `myosuite`               | biomechanics, experimental       | `>=2.0.0`                        | Apache-2.0                            | Commercial-OK          | Optional biomechanics engine; verify bundled model asset terms.              |
| `numpy`                  | core                             | `>=2.0.0`                        | BSD-3-Clause                          | Commercial-OK          | Numerical core.                                                              |
| `opencv-python`          | analysis, pose                   | `>=4.8.0`                        | Apache-2.0                            | Commercial-OK          | Analysis/pose extra; codec patents may apply to deployments.                 |
| `opencv-python-headless` | dev                              | `>=4.13.0.92`                    | Apache-2.0                            | Commercial-OK          | Test video dependency; codec patents may apply to deployments.               |
| `openpose`               | external tool                    | not packaged in `pyproject.toml` | Non-commercial/research-use risk      | Non-commercial, Opt-in | Do not make this the default commercial ingestion path.                      |
| `openpyxl`               | gui-tools                        | `>=3.1.0`                        | MIT                                   | Commercial-OK          | XLSX support.                                                                |
| `opensim`                | biomechanics, experimental       | `>=4.4.0,<5.0.0`                 | Apache-2.0                            | Commercial-OK          | Optional biomechanics engine.                                                |
| `pandas`                 | data, dev, gui-tools             | `>=2.0.0`                        | BSD-3-Clause                          | Commercial-OK          | Dataframe IO/analysis.                                                       |
| `pandera`                | data                             | `>=0.20.0`                       | MIT                                   | Commercial-OK          | Data schema validation.                                                      |
| `pillow`                 | core                             | `>=12.2.0`                       | HPND                                  | Commercial-OK          | Image runtime security floor.                                                |
| `pin`                    | pinocchio                        | `>=2.6.0,<5.0.0`                 | BSD-2-Clause                          | Commercial-OK          | Pinocchio PyPI package.                                                      |
| `platformdirs`           | core                             | `>=4.2.0`                        | MIT                                   | Commercial-OK          | Config/data directory resolution.                                            |
| `pydantic`               | core                             | `>=2.5.0`                        | MIT                                   | Commercial-OK          | Data validation.                                                             |
| `pydantic-settings`      | core                             | `>=2.0.0`                        | MIT                                   | Commercial-OK          | Settings management.                                                         |
| `pyarrow`                | data, dev                        | `>=14.0.0`                       | Apache-2.0                            | Commercial-OK          | Parquet IO.                                                                  |
| `pychrono`               | chrono                           | `>=8.0,<10.0`                    | BSD-3-Clause                          | Commercial-OK          | Optional Project Chrono bindings.                                            |
| `pyjwt`                  | dev                              | `>=2.12.0`                       | MIT                                   | Commercial-OK          | Auth/security tests.                                                         |
| `pyopengl`               | body-part-viz-gl                 | `>=3.1`                          | BSD-3-Clause                          | Commercial-OK          | Optional GL renderer.                                                        |
| `pyproj`                 | core, sg-optimizer               | `>=3.6.0`                        | MIT                                   | Commercial-OK          | Projection support.                                                          |
| `pyqt6`                  | gui-test, gui-tools              | `>=6.5.0`                        | GPL-3.0 OR commercial Qt/PyQt license | Needs review           | Commercial distribution requires an appropriate Qt/PyQt license strategy.    |
| `pyqtgraph`              | body-part-viz-gl                 | `>=0.13`                         | MIT                                   | Commercial-OK          | Optional GL plotting.                                                        |
| `pytest`                 | dev                              | `>=9.0.3`                        | MIT                                   | Commercial-OK          | Test runner.                                                                 |
| `pytest-anyio`           | dev                              | `>=0.0.0`                        | MIT                                   | Commercial-OK          | Async test plugin.                                                           |
| `pytest-asyncio`         | dev                              | `>=1.3.0`                        | Apache-2.0                            | Commercial-OK          | Async test plugin.                                                           |
| `pytest-benchmark`       | dev                              | `>=4.0.0`                        | BSD-2-Clause                          | Commercial-OK          | Benchmark tests.                                                             |
| `pytest-cov`             | dev                              | `>=4.0.0`                        | MIT                                   | Commercial-OK          | Coverage plugin.                                                             |
| `pytest-mock`            | dev                              | `>=3.0.0`                        | MIT                                   | Commercial-OK          | Mocking plugin.                                                              |
| `pytest-qt`              | gui-test                         | `>=4.2.0`                        | MIT                                   | Commercial-OK          | GUI test plugin.                                                             |
| `pytest-timeout`         | dev                              | `>=2.0.0`                        | MIT                                   | Commercial-OK          | Test timeout plugin.                                                         |
| `pytest-xdist`           | dev                              | `>=3.0.0`                        | MIT                                   | Commercial-OK          | Parallel tests.                                                              |
| `python-dotenv`          | core                             | `>=1.2.2`                        | BSD-3-Clause                          | Commercial-OK          | Environment config loading.                                                  |
| `python-multipart`       | core                             | `>=0.0.27`                       | Apache-2.0                            | Commercial-OK          | API multipart parsing.                                                       |
| `pywinpty`               | gui-tools                        | `>=2.0`                          | MIT                                   | Commercial-OK          | Windows PTY support.                                                         |
| `pyyaml`                 | urdf, dev                        | `>=6.0.3`                        | MIT                                   | Commercial-OK          | YAML parsing.                                                                |
| `requests`               | core                             | `>=2.33.0`                       | Apache-2.0                            | Commercial-OK          | HTTP client.                                                                 |
| `robot-descriptions`     | urdf                             | `>=1.12.0`                       | BSD-3-Clause                          | Commercial-OK          | URDF descriptions; verify individual robot asset licenses before bundling.   |
| `ruff`                   | dev                              | `>=0.15.10`                      | MIT                                   | Commercial-OK          | Lint/format.                                                                 |
| `scikit-learn`           | analysis                         | `>=1.3.0`                        | BSD-3-Clause                          | Commercial-OK          | Analysis extra.                                                              |
| `scipy`                  | core                             | `>=1.13.1`                       | BSD-3-Clause                          | Commercial-OK          | Scientific core.                                                             |
| `scipy-stubs`            | dev                              | `>=1.13.0`                       | BSD-3-Clause                          | Commercial-OK          | Type stubs.                                                                  |
| `simpleeval`             | core                             | `>=1.0.0`                        | MIT                                   | Commercial-OK          | Safe expression evaluation.                                                  |
| `slowapi`                | core                             | `>=0.1.9`                        | MIT                                   | Commercial-OK          | API rate limiting.                                                           |
| `sqlalchemy`             | dev                              | `>=2.0.0`                        | MIT                                   | Commercial-OK          | Auth/security tests.                                                         |
| `stable-baselines3`      | rl                               | `>=2.0.0`                        | MIT                                   | Commercial-OK          | RL optional extra.                                                           |
| `structlog`              | core                             | `>=24.1.0`                       | Apache-2.0 OR MIT                     | Commercial-OK          | Structured logging.                                                          |
| `sympy`                  | dev                              | `>=1.12`                         | BSD-3-Clause                          | Commercial-OK          | Symbolic math tests/tools.                                                   |
| `tomli`                  | dev                              | `>=2.0.0`                        | MIT                                   | Commercial-OK          | Python <3.11 TOML parser.                                                    |
| `torch`                  | torch                            | `>=2.0.0`                        | BSD-3-Clause                          | Commercial-OK          | Optional ML dependency; model weights need separate review.                  |
| `trimesh`                | urdf                             | `>=4.11.5`                       | MIT                                   | Commercial-OK          | Mesh processing.                                                             |
| `types-pyyaml`           | dev                              | `>=6.0`                          | Apache-2.0                            | Commercial-OK          | Type stubs.                                                                  |
| `upstream-drift`         | meta extra                       | self-referential extras          | MIT                                   | Commercial-OK          | Extras composition row.                                                      |
| `upstream-mocap-io`      | rust, mocap-io                   | `>=0.1.0`                        | Needs source verification             | Needs review           | Internal/adjacent package; verify package metadata before release.           |
| `upstream-mocap-preproc` | rust, mocap-preproc              | `>=0.1.0`                        | Needs source verification             | Needs review           | Internal/adjacent package; verify package metadata before release.           |
| `upstream-physics`       | rust                             | `>=0.1.0`                        | Needs source verification             | Needs review           | Internal/adjacent package; verify package metadata before release.           |
| `uvicorn`                | core                             | `>=0.30.0`                       | BSD-3-Clause                          | Commercial-OK          | ASGI server.                                                                 |
| `warp-lang`              | warp                             | `>=1.3.0`                        | Apache-2.0                            | Commercial-OK          | Optional NVIDIA Warp dependency.                                             |

## Follow-Up Review Items

- Replace range-based "Version basis" entries with exact versions from the
  production lock once a release lock is designated.
- Verify PyQt6 commercial packaging before distributing GUI bundles.
- Verify robot/model/dataset asset licenses independently from package code
  licenses before bundling sample assets.
- Keep OpenPose external, opt-in, and non-commercial unless legal review
  approves a specific distribution path.
