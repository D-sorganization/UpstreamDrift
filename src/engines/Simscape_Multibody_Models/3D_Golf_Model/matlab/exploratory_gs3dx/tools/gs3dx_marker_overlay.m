function points = gs3dx_marker_overlay(jc, frames)
%GS3DX_MARKER_OVERLAY Joint centres as 3 x targets x selected capture frames.
% JC is already transformed to the address waist frame by capture processing.
    arguments
        jc (1,1) struct
        frames (1,:) double
    end
    names = ["pelvis", "c7", "shoulder_L", "shoulder_R", ...
        "elbow_L", "elbow_R", "wrist_L", "wrist_R", ...
        "hip_L", "hip_R", "knee_L", "knee_R", ...
        "ankle_L", "ankle_R", "toe_L", "toe_R", "club_grip", "club_head"];
    tracks = cell(1, numel(names));
    for i = 1:numel(names)
        tracks{i} = jc.(names(i));
    end
    all_points = permute(cat(3, tracks{:}), [1 3 2]);
    assert(all(isfinite(frames) & frames >= 1 & frames == fix(frames) & frames <= size(all_points, 3)), ...
        'gs3dx:marker_overlay', 'Overlay frames must be valid capture indices');
    points = all_points(:, :, frames);
end
