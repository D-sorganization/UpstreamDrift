# Scapula Matching and Measured-Marker Previews

Moving-scapula whole-body matches use a bilateral protraction prior of 7.5 degrees by default. Address is bounded to 5–10 degrees, and the backswing to 5 degrees through the existing registry limit of 25 degrees. Native primitive signs are mirrored: left Rx is positive protraction; right Rx is negative protraction. Joint roles resolve the native coordinates for each model instead of assuming joint numbers.

The top is the existing measured pelvis-yaw phase proxy, with missing observations excluded. An explicit `scapula_backswing_end_frame` can replace that proxy. The bounds widen with a raised-cosine release between the top and the capture peak-speed proxy. These are matching assumptions; neither scapular anatomy nor ball contact is measured by this policy. Set `scapula_protraction_deg=0` to retain legacy fitting behavior and output layout.

Preserve a selected match's dimensions and offsets when comparing this prior. Review measured-position RMS, head/foot orientation, native torso and pelvis displacement, and frame-to-frame branch continuity together. Reduced torso-hub displacement is a geometric result, not a measurement of upper-body center of mass.

`gs3dx_capture_marker_overlay(cap, frames)` returns every measured capture channel in the same fixed address-waist frame as the joint-centre targets. It keeps residual-invalid and nonfinite samples missing, selects the exact pose frame indices, and reports channel labels and per-frame valid counts. It does not fill missing display markers. The older `gs3dx_marker_overlay(jc, frames)` remains available for joint-centre diagnostics.

`gs3dx_render` defaults to 1920×1080 video frames and MPEG-4 quality 100. Supply `resolution=[width height]` and `video_quality` to override these settings. Dimensions must be even and at least 64 pixels; quality must be 0–100. Still exports use the requested pixel dimensions as well. Video frames use explicit offscreen pixel export, avoiding display-size limits in `getframe`. A renderer request that falls back to GIF does not establish successful MP4 delivery.

For matched Human exports, run both marker modes in the same output directory:

```matlab
for marked = [true false]
    report = gs3dx_match_export("capture-A", ...
        registry_repo=registry_repo, output_dir=output_dir, ...
        views=["face-on","down-the-line"], overlay_markers=marked, ...
        resolution=[1920 1080], video_quality=100);
end
```

The fixed animation bounds include all finite measured markers. Pass the same `scene_markers` array to the renderer in both modes to preserve identical framing.

Marker and clean filenames, stills and provenance files have different suffixes so one mode cannot overwrite the other. Both modes can reuse the same capture/model/geometry/solver-identity checkpoint. Repeat for the owner capture to obtain eight videos. The exporter remains restricted to Human; the shared fitting policy resolves moving-scapula roles across compatible variants.

Before sharing, verify actual decoded dimensions, frame rate, all frame counts and the two-view visual result. Keep private C3D captures, trajectories and owner videos outside Git. These previews demonstrate inverse-kinematics geometry; forward dynamics and contact qualification remain separate work.
