# Review Comments Archive - 2026-05-05

Generated: 2026-05-05T21:54:15.555901

## Reviewer (chatgpt-codex-connector[bot]) (3 comments)

### PR #4022: src/shared/python/motion_matching/loaders/excel.py:47

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Resolve mocap loader path from module location**

`_import_mocap_loader` only searches for `mocap_data_loader.py` relative to `Path.cwd()` and its parents, so `load_club_target_excel` fails with `ImportError` whenever the process runs from a working directory outside the repository tree (for example, a service launched from `/opt` or a notebook started elsewhere). This makes the public loader brittle even whe...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4022#discussion_r3193036540)

---

### PR #4022: src/shared/python/motion_matching/loaders/c3d.py:57

Actionable: No
Has Suggestion: No

```
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Resolve C3D reader path from module location**

`_import_c3d_reader` uses `Path.cwd()` ancestry to locate `c3d_reader.py`, so `load_club_target_c3d` breaks in the same way when callers run outside the repo root hierarchy. In those environments the function raises `ImportError` despite the reader existing in the source tree/package, which blocks C3D ingestion in production-style runtimes.

Useful? React with 👍...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4022#discussion_r3193036544)

---

### PR #4022: src/shared/python/motion_matching/dataset/sweep.py:111

Actionable: Yes
Has Suggestion: No

```
**<sub><sub>![P2 Badge](https://img.shields.io/badge/P2-yellow?style=flat)</sub></sub>  Honor lazy mode without eager timesteps materialization**

`load_sweep_dataset(..., lazy=True)` still reads `timesteps.parquet` eagerly into pandas before returning a LazyFrame, because `pd.read_parquet(timesteps_path)` happens unconditionally. For large sweep datasets this defeats the advertised lazy path and can cause unnecessary memory blowups/OOM during loading; the lazy branch should avoid full eager mat...
```

[View on GitHub](https://github.com/D-sorganization/UpstreamDrift/pull/4022#discussion_r3193036547)

---

