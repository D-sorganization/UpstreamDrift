function matrices = intrinsic_xyz_to_rotm(angles)
%INTRINSIC_XYZ_TO_ROTM World-from-body rotations for intrinsic XYZ radians.
% Rows are independent orientations; output is 3-by-3-by-N. This matches
% KinematicsSolver Rotation variables and applies to column-vector offsets.
    arguments
        angles (:,3) double {mustBeReal, mustBeFinite, mustBeNonempty}
    end
    c = cos(angles); s = sin(angles);
    cx = c(:,1)'; cy = c(:,2)'; cz = c(:,3)';
    sx = s(:,1)'; sy = s(:,2)'; sz = s(:,3)';
    matrices = reshape([cy.*cz; cx.*sz+sx.*sy.*cz; sx.*sz-cx.*sy.*cz; ...
        -cy.*sz; cx.*cz-sx.*sy.*sz; sx.*cz+cx.*sy.*sz; ...
        sy; -sx.*cy; cx.*cy], 3, 3, []);
end
