import json
from pathlib import Path
import numpy as np

repo = Path(__file__).resolve().parent.parent
payload_path = repo / "scratch" / "driver_marker_payload.json"
seed_path = repo / "docs" / "development" / "simscape_tour_matching" / "native_evidence" / "initial_velocity_seed_qualified_r2025b.json"



payload = json.loads(payload_path.read_text(encoding="utf-8"))
seed = json.loads(seed_path.read_text(encoding="utf-8"))


labels_all = payload["labels"]
time_s = np.array(payload["time_s"])
points = np.array(payload["points_world_m"])
valid = np.array(payload["valid"])
points[~valid] = np.nan

print(f"Total frames: {len(time_s)}")
print(f"Time range: [{time_s[0]:.6f}, {time_s[-1]:.6f}] s")
print(f"dt: {time_s[1] - time_s[0]:.6f} s (Hz = {1.0/(time_s[1] - time_s[0]):.1f})")

target_frame = 444
t_target = time_s[target_frame]
print(f"\nFrame {target_frame}: t = {t_target:.6f} s")

assignments = dict(zip(seed["labels"], seed["body_names"], strict=True))
mapped_labels = list(assignments.keys())
mapped_indices = [labels_all.index(l) for l in mapped_labels]

mapped_pts_444 = points[target_frame, mapped_indices]
print(f"\nMapped markers count: {len(mapped_labels)}")

club_indices = [i for i, l in enumerate(labels_all) if any(k in l.lower() for k in ("marker_2", "marker_3", "club"))]
print("\nClub markers in payload:")
for ci in club_indices:
    pos = points[target_frame, ci]
    print(f"  {labels_all[ci]}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m (valid: {valid[target_frame, ci]})")

valid_mapped = mapped_pts_444[~np.isnan(mapped_pts_444).any(axis=1)]
centroid = np.mean(valid_mapped, axis=0)
print(f"\nMapped Marker Centroid at t = {t_target:.4f}s: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}] m")

print("\n--- All Mapped Markers at Frame 444 ---")
for idx, lbl in zip(mapped_indices, mapped_labels):
    pos = points[target_frame, idx]
    body = assignments[lbl]
    print(f"  {lbl:20s} -> {body:20s}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m (valid={valid[target_frame, idx]})")

print("\n--- Marker_2 trajectory and extrapolation around frame 444 ---")
m2_indices = [labels_all.index(f"Marker_2:2:{k}") for k in (1,2,3)]
for f in range(435, 447):
    t = time_s[f]
    pts = points[f, m2_indices]
    v = valid[f, m2_indices]
    if v.all():
        c2 = np.mean(pts, axis=0)
        print(f"  Frame {f:3d} (t={t:.4f}s): M2 Centroid = [{c2[0]:.4f}, {c2[1]:.4f}, {c2[2]:.4f}] m")
    else:
        print(f"  Frame {f:3d} (t={t:.4f}s): M2 invalid (v={v})")


print("\n--- Marker_2 individual markers at frames 443 and 445 ---")
for k in (1, 2, 3):
    idx = labels_all.index(f"Marker_2:2:{k}")
    p443 = points[443, idx]
    p445 = points[445, idx]
    p444_interp = 0.5 * (p443 + p445)
    print(f"  Marker_2:2:{k}:")
    print(f"    f443: [{p443[0]:.4f}, {p443[1]:.4f}, {p443[2]:.4f}] m")
    print(f"    f445: [{p445[0]:.4f}, {p445[1]:.4f}, {p445[2]:.4f}] m")
    print(f"    f444 (interp): [{p444_interp[0]:.4f}, {p444_interp[1]:.4f}, {p444_interp[2]:.4f}] m")

# True Clubhead position at frame 444
p_m2_interp = [0.5 * (points[443, labels_all.index(f"Marker_2:2:{k}")] + points[445, labels_all.index(f"Marker_2:2:{k}")]) for k in (1, 2, 3)]
c_m2_444 = np.mean(p_m2_interp, axis=0)
print(f"\n>>> True Clubhead (Marker_2 Triad Centroid) at t = {t_target:.4f}s: [{c_m2_444[0]:.4f}, {c_m2_444[1]:.4f}, {c_m2_444[2]:.4f}] m")

# Marker_3 centroid at frame 444 (on shaft)
m3_pts = points[target_frame, [labels_all.index(f"Marker_3:3:{k}") for k in (1,2,3)]]
c_m3_444 = np.mean(m3_pts, axis=0)
print(f">>> Shaft Marker Triad (Marker_3 Centroid) at t = {t_target:.4f}s: [{c_m3_444[0]:.4f}, {c_m3_444[1]:.4f}, {c_m3_444[2]:.4f}] m")

# Clubhead velocity around frame 444:
v_head = (np.mean(points[445, m2_indices], axis=0) - np.mean(points[443, m2_indices], axis=0)) / (2 * (1.0/360.0))
speed_head_mps = np.linalg.norm(v_head)
speed_head_mph = speed_head_mps * 2.23694
print(f"\n>>> Clubhead Velocity Vector: [{v_head[0]:.2f}, {v_head[1]:.2f}, {v_head[2]:.2f}] m/s")
print(f">>> Clubhead Speed: {speed_head_mps:.2f} m/s ({speed_head_mph:.1f} mph = {speed_head_mps*3.6:.1f} km/h)")

# Shaft Vector: from Clubhead Centroid to Mid-Wrist or to Marker_3 (Shaft)
lw = points[target_frame, labels_all.index("LWristTop")]
rw = points[target_frame, labels_all.index("RWristTop")]
mid_wrist = 0.5 * (lw + rw)

# Full shaft axis: Mid-Wrist -> Clubhead
full_shaft_vec = c_m2_444 - mid_wrist
full_shaft_len = np.linalg.norm(full_shaft_vec)
full_shaft_u = full_shaft_vec / full_shaft_len

# Shaft segment axis: Marker_3 -> Clubhead
shaft_segment_vec = c_m2_444 - c_m3_444
shaft_segment_len = np.linalg.norm(shaft_segment_vec)
shaft_segment_u = shaft_segment_vec / shaft_segment_len

print(f"\n>>> Hands / Mid-Wrist: [{mid_wrist[0]:.4f}, {mid_wrist[1]:.4f}, {mid_wrist[2]:.4f}] m")
print(f">>> Full Shaft Vector (Hands -> Clubhead): [{full_shaft_vec[0]:.4f}, {full_shaft_vec[1]:.4f}, {full_shaft_vec[2]:.4f}] m (Length: {full_shaft_len:.4f} m)")
print(f">>> Full Shaft Direction Unit Vector: [{full_shaft_u[0]:.4f}, {full_shaft_u[1]:.4f}, {full_shaft_u[2]:.4f}]")
print(f">>> Shaft Segment Vector (Marker_3 -> Clubhead): [{shaft_segment_vec[0]:.4f}, {shaft_segment_vec[1]:.4f}, {shaft_segment_vec[2]:.4f}] m (Length: {shaft_segment_len:.4f} m)")
print(f">>> Shaft Segment Direction Unit Vector: [{shaft_segment_u[0]:.4f}, {shaft_segment_u[1]:.4f}, {shaft_segment_u[2]:.4f}]")

# Delivery Orientation Angles:
# 1. Shaft Lean (forward/backward shaft lean in swing plane):
# In golf biomechanics: forward lean angle relative to vertical:
shaft_lean_from_vert = np.degrees(np.arccos(abs(full_shaft_u[2])))
shaft_inclination = np.degrees(np.arcsin(abs(full_shaft_u[2])))
shaft_azimuth = np.degrees(np.arctan2(full_shaft_u[1], full_shaft_u[0]))
print(f"\n>>> Shaft Delivery Orientation:")
print(f"    Inclination from Horizontal: {shaft_inclination:.2f} deg")
print(f"    Lean from Vertical:          {shaft_lean_from_vert:.2f} deg")
print(f"    Azimuth in X-Y Plane:        {shaft_azimuth:.2f} deg")



