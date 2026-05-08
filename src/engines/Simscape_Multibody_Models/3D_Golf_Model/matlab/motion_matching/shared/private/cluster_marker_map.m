function map = cluster_marker_map()
%CLUSTER_MARKER_MAP  Validated cluster-marker C3D convention.
%
%   MAP = CLUSTER_MARKER_MAP() returns a struct describing the marker schema
%   for the cluster-marker mocap files in
%       src/.../matlab/Data/Mocap C3D Files/
%   that were validated externally via ezc3d 1.7.0 (see PR #3982 follow-up).
%
%   Fields:
%     clubhead_cluster (1x3 string) -- the three rigid-cluster markers on the
%                                      clubhead (Marker_2:2:{1,2,3}).
%     grip_cluster     (1x3 string) -- the three rigid-cluster markers on the
%                                      grip / butt (Marker_3:3:{1,2,3}).
%     sentinel_markers (1xS string) -- markers carrying stuck values.
%     occluded_markers (1xO string) -- markers known-occluded across the
%                                      validated traces.
%     excluded_markers (1xE string) -- union of sentinels and occluded
%                                      markers.  Never use for fitting.
%     max_gap_frames   scalar       -- longest NaN gap (frames) eligible for
%                                      spline interpolation.
%     source_units     string       -- "m" (metres at the source).
%     source_axes      string       -- "Y-up" (Vicon convention).
%     world_axes       string       -- "Z-up" (Simscape convention; the
%                                      loader applies (x,y,z) -> (x,-z,y)).
    map = struct( ...
        "clubhead_cluster", ["Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3"], ...
        "grip_cluster",     ["Marker_3:3:1", "Marker_3:3:2", "Marker_3:3:3"], ...
        "sentinel_markers", "Marker_0:0:0", ...
        "occluded_markers", "RShoulderTop", ...
        "excluded_markers", ["Marker_0:0:0", "RShoulderTop"], ...
        "max_gap_frames",   5, ...
        "source_units",     "m", ...
        "source_axes",      "Y-up", ...
        "world_axes",       "Z-up");
end
