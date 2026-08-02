# Motion-capture parity: web is MediaPipe-only; desktop ships C3D viewer + OpenPose + MediaPipe

## Gap (PyQt6 = model)

The desktop Motion Capture tile fans out to three tools (`src/launchers/motion_capture_launcher.py`): C3D viewer (`src/engines/Simscape_Multibody_Models/3D_Golf_Model/python/src/apps/c3d_viewer.py` + Rust loaders), OpenPose, and MediaPipe. The web `MotionCapture` page (`ui/src/pages/MotionCapture.tsx`) only exercises the MediaPipe-style flow. The backend already exposes generic capture endpoints (`/capture/sources`, `/capture/skeleton/{source}`, `/capture/session/*`, `/capture/recordings`, `/capture/playback` in `src/api/routes/motion_capture.py`) — the web page just doesn't cover the other sources, and C3D file upload/inspection has no web path.

## Proposed fix

1. Drive the web source selector entirely from `GET /capture/sources` (it already lists what the backend supports) — remove any hardcoded source list.
2. C3D: add upload (reuse the multipart pattern from `apiFetchForm`) → parse via the existing C3D loaders → marker/skeleton playback in the existing 2D/3D visualizer; expose marker metadata (units fix from #7200 lineage applies — verify unit handling in the web render path).
3. OpenPose: wire the source through the same session/playback endpoints; if OpenPose isn't installed server-side, the source must report unavailable (honest capability), not silently fall back.
4. Skeleton definitions per source come from `GET /capture/skeleton/{source}` so desktop and web render the same joint sets.

## Acceptance criteria

- [ ] All capture sources reported by the backend are usable from the web page; unavailable ones shown as such with reason
- [ ] A C3D file can be uploaded, inspected (markers/rate/duration), and played back in the web 3D view
- [ ] No hardcoded source/skeleton lists in `ui/`
- [ ] Parity test: web-consumed source list == desktop sub-launcher tool list (via shared backend enumeration)

## References

- `docs/development/launcher_parity_assessment.md` §9 (the three tools are one tile by design)
- Closed #1173 (MediaPipe/OpenPose GUIs were mock-only — verify estimator wiring state before building on it)
