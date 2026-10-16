# Open Source Integration Scout — Weekly Report (2026-10-16)

**Role:** Open Source Integration Scout (`oss-scout`), per
[`fleet-oss-scout.md`](https://github.com/D-sorganization/Repository_Management/blob/main/docs/fleet-oss-scout.md).

**Disposition:** Guidance only. No issues filed, no leases taken, no
dependencies merged, no code changed outside this report.

**Scope this pass:** UpstreamDrift only (`main` @
`a64016d6e18e242f9fac13916afc1c50c24e8b5c`), triggered by two gaps already
flagged in-repo: the `FBX` and `Qualisys (.qtm)` rows marked `_planned_` in
[`docs/motion_pipeline/formats.md`](../motion_pipeline/formats.md), plus one
dependency-hygiene gap found while reading `src/shared/python/realtime/`.

## Candidate 1 — FBX Motion-Capture Ingestion (`pufbx` / `pyufbx`): Reject for Now

1. **Use case:** `docs/motion_pipeline/formats.md` lists an FBX adapter as
   `_planned_` with no implementation under
   `src/shared/python/motion_pipeline/sources/`. A native Python FBX reader
   would let the motion pipeline ingest FBX exports directly instead of
   requiring an external conversion step.
2. **Alternatives already in the stack:** none directly — BVH, TRC, C3D,
   STO/MOT and five pose-estimator JSON formats are covered, but no FBX path
   exists today, planned or otherwise wired to `load_any`.
3. **License:** both Python bindings wrap the `ufbx` C library
   (<https://github.com/ufbx/ufbx>, MIT) under MIT:
   - `pufbx` 0.1.0 (released 2026-04-19,
     <https://github.com/Owca-Xsido/pufbx>, single maintainer, Cython
     wrapper).
   - `pyufbx` 0.0.7 (released 2026-01-26,
     <https://github.com/popomore/ufbx-python>, no named maintainer beyond
     "ufbx-python contributors").
4. **Maintenance / bus-factor:** both bindings are **under a year old**,
   pre-1.0, and effectively bus-factor-1 (single author or an unnamed
   contributor group, one release train each). The underlying `ufbx` C
   library itself is mature and widely consumed (it backs Blender's own FBX
   importer since Blender 4.x), but neither Python wrapper has the release
   history or contributor count the playbook asks for before adoption.
5. **Security / supply chain:** `ufbx` is a single-file C library with no
   transitive dependencies, which is a good sign; both Python wrappers add a
   compiled Cython extension, which means wheels must exist for every CI
   platform (Linux/macOS/Windows × CPython 3.11/3.12) or the build falls back
   to a local C/Cython toolchain in CI.
6. **API / stability risk:** pre-1.0 versioning (`0.1.0`, `0.0.7`) on both —
   no stability guarantee, and no evidence yet of a 1.0 or a deprecation
   policy.
7. **Integration cost:** small if the wrapper's API is stable in practice —
   roughly a day to write an `FBXAdapter` mirroring the existing
   `BVHAdapter`/`TRCAdapter` shape plus golden-fixture tests. The risk isn't
   the adapter code, it's depending on a sub-1.0, single-maintainer compiled
   wheel for a format that isn't blocking any current epic.
8. **Reversible PoC (not run this pass):** vendor one `.fbx` fixture, write
   `FBXAdapter` behind the existing `registry.py` sniffing contract using
   `pufbx` (more actively released of the two), and run it only inside the
   optional `mocap-io`/`c3d`-style extra so core installs are unaffected.
   Unwind: delete the adapter module, its registry entry and the extra
   stanza — no other module would reference it given the project's existing
   lazy-import-at-call-site pattern (see `ws_pubsub.py` for precedent).
9. **Exit plan if adopted then abandoned:** FBX is an interchange format, not
   a data of record — nothing in the pipeline would depend on `ufbx` for
   correctness once ingested into the canonical pose format, so dropping the
   adapter later only removes the one entry point.

**Verdict: reject for this pass.** Neither binding has the track record this
portfolio's playbook asks for, and no current epic in `AGENT_HANDOFF.md` is
blocked on FBX ingestion. Revisit in 2–3 quarters once `pufbx` (the more
actively released of the two) has a 1.0 and a second release past its first.
If FBX support becomes urgent before then, the lower-risk path is a
subprocess call into an external converter — the same pattern this repo
already uses for the HMR2 sidecar
(`src/tools/hmr2_sidecar/run_hmr2.py`) to avoid importing
licensing-encumbered code directly — rather than taking a pre-1.0 compiled
dependency into the core pipeline.

## Candidate 2 — Qualisys QTM Native-Format Parsing (`qtm` / `qtm-rt`): Reject, Close the Gap Differently

1. **Use case:** the same format matrix row lists a native `.qtm` adapter as
   `_planned_`, implying a desire to read Qualisys Track Manager's project
   format directly.
2. **Alternatives already in the stack:** `C3DAdapter`
   (`src/shared/python/motion_pipeline/sources/c3d_adapter.py`, `ezc3d`-backed)
   and `TRCAdapter` already cover Qualisys output — QTM natively exports both
   C3D and TRC, which is the documented, supported interchange path for this
   vendor, not the proprietary `.qtm` project file.
3. **License / maintenance:** two Qualisys-published SDKs exist on PyPI:
   - `qtm` 2.1.2 (released 2023-06-09,
     <https://github.com/qualisys/qualisys_python_sdk>) — PyPI metadata
     marks it **deprecated** in favor of `qtm-rt`.
   - `qtm-rt` 3.1.0 (released 2026-06-10, same repo) — actively maintained,
     MIT.
   Neither parses the `.qtm` project file; both implement the **real-time
   streaming protocol** (live TCP/UDP capture from a running QTM session),
   which is a different problem (live capture ingestion) from the file-based
   motion pipeline this matrix row is part of.
4. **API / stability risk / integration cost:** not applicable — wrong tool
   for the stated use case, so there's nothing to integrate.
5. **Reversible PoC:** none proposed.
6. **Exit plan:** none needed.

**Verdict: reject**, and recommend the `_planned_` row be resolved by
documentation rather than new code: change it to point readers at the
existing `C3DAdapter`/`TRCAdapter` path (Qualisys's own documented export
route) instead of implying a native parser is coming. If live-streaming
ingestion from a running QTM session is ever actually wanted, `qtm-rt`
3.1.0 is a legitimate, well-maintained candidate for *that* separate
use case — but it should be scoped as its own decision (likely a
`realtime`-package integration, not a `motion_pipeline` file adapter), not
folded into closing this row.

## Candidate 3 — `deptry` for Dependency Hygiene: Recommend a Scoped PoC, Not Adopted This Pass

1. **Use case:** while reading `src/shared/python/realtime/ws_pubsub.py` for
   the Qualisys streaming comparison above, I found `import websockets`
   inside a function body (`ws_pubsub.py:336`) with no corresponding
   `websockets` entry anywhere in `pyproject.toml` — not in core
   dependencies, not in any extra. It degrades gracefully if missing (lazy
   import), so it isn't a live bug, but it is exactly the class of drift
   `deptry` exists to catch: an import with no declared dependency, which
   would otherwise only surface the next time someone tightens the extras.
2. **Alternatives already in the stack:** `pip-audit` already runs in CI
   (`.github/workflows/ci-standard.yml`, `.github/workflows/release.yml`) —
   that covers known-vulnerability scanning, not unused/missing/transitive
   dependency drift, which is a different failure mode.
3. **License:** MIT.
4. **Maintenance / bus-factor:** `deptry` 0.25.1, maintained by Florian Maas
   and Mathieu Kniewallner, actively released, multi-year history, supports
   pip/PEP 621 projects (this repo's `pyproject.toml` shape) directly.
5. **Security / supply chain:** pure dev-time static-analysis tool, not a
   runtime dependency — it never ships in production installs.
6. **API / stability risk:** low — CLI tool invoked once per CI run, not
   imported by application code.
7. **Integration cost:** small, order of an hour: add it as a dev-only
   dependency, run `deptry src` once locally to baseline existing findings
   (this repo's extras-heavy `pyproject.toml` will likely need a config
   allowlist for intentionally-lazy optional imports like the `websockets`
   case above, and for the `rust`/`mocap-*` extras that are satisfied by
   Rust wheels rather than pure-Python imports).
8. **Reversible PoC (not run this pass):** add `deptry` to a dev-only extra,
   run it locally against `src/` to produce a baseline report, and file
   *one* follow-up issue (not opened this pass) with the findings if the
   baseline is small enough to be useful. Unwind: drop the dev dependency;
   nothing in `src/` would reference it.
9. **Exit plan if adopted then abandoned:** zero runtime exposure — it only
   ever runs as a CI/local lint step, so dropping it is a one-line revert of
   the dev-dependency declaration and any CI step.

**Verdict: worth a follow-up, not actioned this pass.** This role holds no
permissions to add dev dependencies or file issues; recommending it to
Orchestrator as a cheap, reversible CI addition, with the undeclared
`websockets` import as the concrete first finding it would have caught.

## Handoff

Sent to Orchestrator first, per playbook. No `staff-actions` block: this
role's `permissions`/`tools` authorize no issue filing, lease posting, or
dependency merge, so none is proposed here — Orchestrator or Dieter would
need to gate any of the three follow-ups above (an `FBXAdapter` PoC behind
an extra, a `motion_pipeline/formats.md` doc fix for the QTM row, or a
`deptry` dev-dependency PoC) before an implementation agent picks them up.

```text
handoff: orchestrator
question: Is FBX ingestion or QTM native parsing actually blocking any
  active epic in AGENT_HANDOFF.md right now, or is the formats.md
  "_planned_" marking stale and safe to resolve as docs-only?
```
