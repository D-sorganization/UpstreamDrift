function [T, ang, qd, qdd] = gs3dx_xyz_map(Q, w, b, tau, damping, ang_ref)
%GS3DX_XYZ_MAP  Spherical joint state <-> Gimbal (intrinsic X-Y-Z) quantities.
%
%   [T, ANG, QD, QDD] = GS3DX_XYZ_MAP(Q, W, B, TAU, DAMPING, ANG_REF) lets a
%   Spherical Joint stand in for a Gimbal Joint (#10955).  Inputs:
%     Q        quaternion [w x y z] of the follower relative to the base
%     W, B     follower angular velocity (rad/s) and acceleration (rad/s^2)
%              relative to the base, resolved in the follower frame
%     TAU      Gimbal axis torques [tx ty tz] (N*m)
%     DAMPING  Gimbal axis damping coefficients (N*m/(deg/s))
%     ANG_REF  continuous reference angles (deg), e.g. integrated rates,
%              used only to pick the 360-degree branch of a and c
%   Outputs (all 3x1):
%     T    follower-frame torque with the same power as the Gimbal:
%          T = E^-T * (TAU - DAMPING .* QD), E = GS3DX_XYZ_RATE_MATRIX
%     ANG  angles [a b c] (deg), a and c unwrapped to the branch of ANG_REF
%     QD   angle rates (deg/s) = E \ W
%     QDD  angle accelerations (deg/s^2) = E \ (B - dE/dt * QD)
%
%   Assumes cos(b) > 0 (|b| < 90 deg), the range the Gimbal's middle angle
%   stays in; the map is singular at gimbal lock.  Code-generation compatible.

    q = Q(:) / norm(Q);
    R = local_rotm(q);
    sb = R(1, 3);
    cb = hypot(R(1, 1), R(1, 2));
    a_w = atan2(-R(2, 3), R(3, 3));
    b_r = atan2(sb, cb);
    c_w = atan2(-R(1, 2), R(1, 1));
    sc = sin(c_w); cc = cos(c_w);

    E = gs3dx_xyz_rate_matrix(b_r, c_w);
    qd_r = E \ w(:);
    da = qd_r(1); db = qd_r(2); dc = qd_r(3);
    Edot_qd = da * [-sb * cc * db - cb * sc * dc; ...
                     sb * sc * db - cb * cc * dc; ...
                     cb * db] ...
            + db * [cc * dc; -sc * dc; 0];
    qdd_r = E \ (b(:) - Edot_qd);

    ref = ang_ref(:) * pi / 180;
    ang_r = [local_unwrap(a_w, ref(1)); b_r; local_unwrap(c_w, ref(3))];

    qd  = qd_r * 180 / pi;
    qdd = qdd_r * 180 / pi;
    ang = ang_r * 180 / pi;
    T = E.' \ (tau(:) - damping(:) .* qd);
end

function R = local_rotm(q)
% Rotation matrix of unit quaternion q = [w x y z] (follower -> base).
    w = q(1); x = q(2); y = q(3); z = q(4);
    R = [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y); ...
         2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x); ...
         2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)];
end

function x = local_unwrap(x_wrapped, ref)
    x = x_wrapped + 2 * pi * round((ref - x_wrapped) / (2 * pi));
end
