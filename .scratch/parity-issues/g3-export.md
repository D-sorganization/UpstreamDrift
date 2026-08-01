# Export/recording parity: desktop has 5 formats + persisted recordings, web has JSON-only and no recording access

## Gap (PyQt6 = model)

Desktop (`GenericPhysicsRecorder`, `src/shared/python/dashboard/recorder.py`) records full simulation sessions and exports **HDF5, CSV, MAT, C3D, JSON** (plus video via `video_export.py` on MuJoCo/Simscape). The web app can only retrieve `GET /export/{task_id}` JSON; the CSV path in the API is a stub, and HDF5/MAT/C3D/video are not exposed at all. Recordings written to `output/` are invisible to the web client.

## Proposed fix

1. **Recording persistence + listing API**:
   - `POST /api/v1/recordings` (finalize current session recorder to disk), `GET /api/v1/recordings` (list persisted recordings with metadata), `DELETE /api/v1/recordings/{id}`.
   - Storage under the existing `output/` convention; index with metadata (engine, model, duration, created).
2. **Binary export download**: `GET /api/v1/recordings/{id}/export?format=hdf5|csv|mat|c3d|json` streaming the file (FastAPI `FileResponse`), implemented by calling the _same_ recorder export methods the desktop uses — no parallel serialization code. `GET /api/v1/export/formats` enumerates formats (already proposed in the dual-GUI review) so the web UI is data-driven.
3. **Web UI**: Recordings panel (list + download per format + delete) on the Simulation/Analysis pages; wire the existing-but-dead "trajectory export" affordance in `SimulationControls.tsx` to this path.
4. Complete or delete the CSV stub flagged in the stub-endpoints issue of this epic.
5. Video export: expose for engines whose capability profile advertises it (MuJoCo today) as an async task producing a downloadable MP4; gate by capabilities, don't fake for others.

## Acceptance criteria

- [ ] Same byte-identical export files obtainable from web and desktop for the same recording (golden test on a short pendulum run)
- [ ] Formats enumerated by API == formats offered by desktop recorder (parity test)
- [ ] Recordings listable/downloadable/deletable from the web UI
- [ ] No new serialization code paths — desktop and API share the recorder implementation

## References

- `docs/architecture/dual-gui-architecture-review.md` §3.2 Step 2 (recording/export endpoints)
- `src/api/routes/export.py`, `src/api/routes/dataset.py`
- Closed #1176 (video/dataset export from all engines) — verify what actually landed before scoping video work
